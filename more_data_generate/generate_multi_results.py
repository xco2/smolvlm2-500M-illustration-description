import os
import json
import random
import time
import yaml

import tqdm
import sys
import signal
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from multiprocessing import Process, Queue

# 加载配置文件
def load_config(config_path="multi_results_config.yaml"):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 加载配置
config = load_config()

# 从配置中获取路径
input_jsonl_path = config["paths"]["input_jsonl"]
output_jsonl_path = config["paths"]["output_jsonl"]
image_dir = config["paths"]["image_dir"]

# 从配置中获取模型配置
MODEL_ID = config["model"]["model_id"]
step = config["model"]["step"]
LORA_PATH = config["model"]["lora_path_template"].format(step=step)

# 从配置中获取生成配置
batch_size = config["generation"]["batch_size"]
generate_data_num = config["generation"]["generate_data_num"]

# 定义一个标志变量，用于指示是否收到停止信号
stop_requested = False


def signal_handler(sig, frame):
    global stop_requested
    stop_requested = True
    print("收到停止信号，将在保存完当前数据后退出...")


# 注册信号处理函数
signal.signal(signal.SIGINT, signal_handler)


def load_jsonl(file_path):
    """加载jsonl文件并返回数据列表"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    except Exception as e:
        print(f"加载数据失败: {e}")
        return []


def save_jsonl(data, file_path):
    """保存数据到jsonl文件"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_model_and_processor():
    """加载模型和处理器"""
    print("加载模型和处理器...")
    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    if __name__ == "__main__":
        # 加载基础模型
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=getattr(torch, config["model"]["torch_dtype"]),
            _attn_implementation=config["model"]["attn_implementation"]
        )

        model = PeftModel.from_pretrained(model, LORA_PATH)

        # 可选: 合并LoRA权重到基础模型。
        model = model.merge_and_unload().eval()
        # 将最终的模型移动到CUDA设备
        model = model.to("cuda").to(getattr(torch, config["model"]["torch_dtype"]))
    else:
        model = None

    return processor, model


def merge_inputs(instances, dtype):
    input_ids = pad_sequence(
        [inst["input_ids"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id,
        padding_side='left'
    ).to("cuda")
    attention_mask = pad_sequence(
        [inst["attention_mask"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=0,
        padding_side='left'
    ).to("cuda")

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    # Step 1: figure out maximum frames, height, width across the batch
    pvs = [inst["pixel_values"].squeeze(0) for inst in instances if "pixel_values" in inst]
    if pvs:  # there is at least one non-None pixel_values
        max_frames = max(pv.shape[0] for pv in pvs)
        max_h = max(pv.shape[-2] for pv in pvs)
        max_w = max(pv.shape[-1] for pv in pvs)
    else:
        max_h = max_w = processor.video_size['longest_edge']
        max_frames = 1

    padded_pixel_values_list = []
    padded_mask_list = []
    for ex in instances:
        pv = ex.get("pixel_values", None).squeeze(0)

        if pv is None:
            # text-only => fill pixel data + mask with zeros
            shape_pv = (max_frames, 3, max_h, max_w)
            padded_pv = torch.zeros(shape_pv, device=input_ids.device, dtype=dtype)
            padded_mask = torch.zeros(shape_pv[:2], device=input_ids.device, dtype=torch.int64)
        else:
            f, c, h, w = pv.shape
            # Prepare final storage
            padded_pv = torch.zeros(
                (max_frames, c, max_h, max_w),
                dtype=pv.dtype,
                device=pv.device
            )
            padded_mask = torch.zeros(
                (max_frames, max_h, max_w),
                dtype=torch.int64,
                device=pv.device
            )
            padded_pv[:f, :, :h, :w] = pv
            padded_mask[:f, :h, :w] = 1
        padded_pixel_values_list.append(padded_pv)
        padded_mask_list.append(padded_mask)

    inputs["pixel_values"] = torch.stack(padded_pixel_values_list, dim=0).to("cuda").to(getattr(torch, config["model"]["torch_dtype"]))
    inputs["pixel_attention_mask"] = torch.stack(padded_mask_list, dim=0).to("cuda")
    return inputs


def generate_results(image_paths, prompts):
    """使用模型生成多种不同的结果"""

    batch_messages = []
    for image_path, prompt in zip(image_paths, prompts):
        # 准备输入
        user_content = [{"type": "image", "url": image_path},
                        {"type": "text", "text": prompt}]

        messages = [
            {"role": "user", "content": user_content}
        ]
        batch_messages.append(messages)

    n_data = len(batch_messages)
    st = time.time()
    inputs = processor.apply_chat_template(
        batch_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to("cuda").to(getattr(torch, config["model"]["torch_dtype"]))
    print(f"processor: {time.time() - st:.4f}s")

    # 从配置中获取生成策略
    generation_results = {}
    
    with torch.no_grad():
        # 遍历所有启用的生成策略
        for strategy_key, strategy_config in config["generation_strategies"].items():
            # 复制策略配置并移除name字段，因为它不是generate方法的参数
            generate_kwargs = {k: v for k, v in strategy_config.items() if k != "name"}
            
            # 使用当前策略生成结果
            generated_ids = model.generate(
                **inputs,
                **generate_kwargs
            )
            
            # 解码生成的文本
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 保存这个策略的结果
            strategy_name = strategy_config["name"]
            generation_results[strategy_name] = generated_texts

    # 去除生成文本中的提示部分
    assistant_prefix = "Assistant:"
    res = []
    for i in range(n_data):
        result_item = {}
        
        # 处理每个策略的结果
        for strategy_name, texts in generation_results.items():
            text = texts[i]
            if assistant_prefix in text:
                text = text.split(assistant_prefix)[1]
            result_item[strategy_name] = text
            
        res.append(result_item)
        
    return res


# 加载模型和处理器
processor, model = load_model_and_processor()


def main():
    global stop_requested
    print("加载数据...")
    selected_data_path = os.path.join(os.path.dirname(output_jsonl_path), config["paths"]["selected_data"])
    progress_path = os.path.join(os.path.dirname(output_jsonl_path), config["paths"]["progress"])

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)

    # 加载已选中的数据（如果存在）
    if os.path.exists(selected_data_path):
        print("加载已保存的抽样数据...")
        data = load_jsonl(selected_data_path)
    else:
        # 原抽样逻辑
        # data = load_jsonl(input_jsonl_path)
        # if len(data) > generate_data_num:  # 只处理500个样本
        #     data = random.sample(data, generate_data_num)
        # # 保存抽样结果
        # print("保存抽样数据...")
        # save_jsonl(data, selected_data_path)
        raise ValueError("找不到已保存的抽样数据")

    data_len = len(data)

    # 加载处理进度
    start_index = 0
    if os.path.exists(progress_path):
        with open(progress_path, 'r', encoding='utf-8') as f:
            progress = f.read().strip()
            if progress.isdigit():
                start_index = int(progress)
    print(f"当前处理进度：已完成 {start_index}/{data_len}，继续从索引 {start_index} 开始处理...")

    # 加载已有结果（如果存在）
    if os.path.exists(output_jsonl_path):
        results = load_jsonl(output_jsonl_path)
        print(f"已加载 {len(results)} 条已有数据。")
    else:
        results = []

    print("开始生成多样化结果...")
    tqdm_total = data_len // batch_size if data_len % batch_size == 0 else data_len // batch_size + 1
    for i in tqdm.tqdm(range(start_index, data_len, batch_size), initial=start_index // batch_size, total=tqdm_total):
        items = data[i:i + batch_size]

        # 获取图片路径和提示
        image_paths = []
        prompts = []
        for item in items:
            img_path = os.path.join(image_dir, item["image"].replace("\\", "/"))
            prompt = item["conversations"][0]["content"].replace("\n<image>", "")
            # 去除问题中的字数限制提示
            if prompt.startswith("Write"):
                prompt = prompt.replace("100 words ", "")
            elif prompt.startswith("Analyze"):
                prompt = prompt.replace(" Keep it 100 words.", "")

            # 检查图片是否存在
            if not os.path.exists(img_path):
                print(f"图片不存在: {img_path}，跳过")
                continue

            image_paths.append(img_path)
            prompts.append(prompt)

        # 生成结果
        generated_results = generate_results(image_paths, prompts)

        for j, res in enumerate(generated_results):
            # 保存结果
            result_item = {
                "image": items[j]["image"],
                "prompt": prompts[j],
                "original_answer": items[j]["conversations"][1]["content"],
                "generated_results": res
            }

            results.append(result_item)

        # 更新进度
        with open(progress_path, 'w', encoding='utf-8') as f:
            f.write(str(i + batch_size))

        # 定期保存
        if (i + 1) % 5 == 0:
            print(f"已生成 {i + batch_size} 条数据，正在保存...")
            save_jsonl(results, output_jsonl_path)

        # 检查是否收到停止信号
        if stop_requested:
            print("收到停止信号，正在保存当前进度...")
            save_jsonl(results, output_jsonl_path)
            print(f"已保存当前进度，当前进度 {i + batch_size}/{data_len}")
            sys.exit(0)

    # 最终保存
    print(f"已生成 {len(results)} 条数据，正在保存...")
    save_jsonl(results, output_jsonl_path)
    print(f"完成！共生成 {len(results)} 条数据")


if __name__ == "__main__":
    main()
