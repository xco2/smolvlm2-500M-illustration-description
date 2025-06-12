# import unsloth
# from unsloth import FastLanguageModel
import os
import torch
from torch import nn
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import yaml
from transformers import (
    AutoProcessor,
    Trainer,
    AutoModelForImageTextToText
)
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel, get_peft_model
from trl import GRPOConfig
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# ========== 加载配置文件 ==========
def load_config(config_path="grpo_config.yaml"):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


config = load_config()

# ========== 配置部分 ==========
USE_LORA = config["use_lora"]  # 是否使用LoRA微调
SMOL = config["smol"]  # 是否使用小模型
MODEL_ID = config["model_id"] if SMOL else config["model_id_large"]

# ========== 训练参数 ==========
model_name = MODEL_ID.split("/")[-1]
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["WANDB_PROJECT"] = config["wandb"]["project"]
if config["wandb"]["mode"] == "online":
    os.environ["WANDB_API_KEY"] = config["wandb"]["api_key"]
else:
    os.environ["WANDB_MODE"] = config["wandb"]["mode"]

# --------LoRA 配置----------
if SMOL:
    lora_config = config["smol_lora"]
    target_modules_layer = [i for i in range(lora_config["target_modules_layer"])]  # 要进行 LoRA 微调的层索引
    LORA_R = lora_config["lora_r"]  # LoRA 中的 R 参数
    LM_HEAD_RANk = lora_config["lm_head_rank"]  # lm_head层的rank
    CONNECT_RANK = lora_config["connect_rank"]  # connector层的rank

    target_modules_lora_r = lora_config["target_modules_lora_r"]  # 每个层的 R 参数
    LORA_RANKS = {layer_idx: lora_r for layer_idx, lora_r in zip(target_modules_layer, target_modules_lora_r)}

    epoch = lora_config["epoch"]
    batch_size = lora_config["batch_size"]
    lr = lora_config["lr"]
    gradient_steps = lora_config["gradient_steps"]
else:
    lora_config = config["large_lora"]
    target_modules_layer = [i for i in range(lora_config["target_modules_layer"])]  # 要进行 LoRA 微调的层索引
    LORA_R = lora_config["lora_r"]  # LoRA 中的 R 参数
    LM_HEAD_RANk = lora_config["lm_head_rank"]  # lm_head层的rank
    CONNECT_RANK = lora_config["connect_rank"]  # connector层的rank

    # 对于大模型，使用动态计算每层的R参数
    target_modules_lora_r = lora_config["target_modules_lora_r"]  # 每个层的 R 参数
    LORA_RANKS = {layer_idx: lora_r for layer_idx, lora_r in zip(target_modules_layer, target_modules_lora_r)}

    epoch = lora_config["epoch"]
    batch_size = lora_config["batch_size"]
    lr = lora_config["lr"]
    gradient_steps = lora_config["gradient_steps"]

# 训练数据集路径
dataset_path = config["dataset"]["train_path"]
test_dataset_path = config["dataset"]["test_path"]

# 图片路径
image_dir = config["dataset"]["image_dir"]
bf_16 = config["training"]["bf16"]  # 是否使用bf16训练

RANDOM_SEED = config["training"]["random_seed"]

training_args = GRPOConfig(
    num_train_epochs=epoch,
    use_vllm=config["training"]["use_vllm"],  # 使用vLLM加速推理
    learning_rate=lr,  # 学习率
    warmup_steps=config["training"]["warmup_steps"],
    adam_beta1=config["training"]["adam_beta1"],  # Adam优化器参数
    adam_beta2=config["training"]["adam_beta2"],
    weight_decay=config["training"]["weight_decay"],  # 权重衰减
    lr_scheduler_type=config["training"]["lr_scheduler_type"],  # 学习率调度策略
    lr_scheduler_kwargs={"min_lr": config["training"]["lr_min"]},
    optim=config["training"]["optim"],
    logging_steps=config["training"]["logging_steps"],
    bf16=bf_16,  # 使用已定义的bf_16变量
    per_device_train_batch_size=batch_size,  # batch size
    gradient_accumulation_steps=gradient_steps,  # 累计n步后更新一次参数
    num_generations=batch_size,  # 每次生成的候选数
    max_prompt_length=config["training"]["max_prompt_length"],  # 输入最大长度
    max_completion_length=config["training"]["max_completion_length"],  # 生成最大长度
    max_steps=config["training"]["max_steps"],  # 最大训练步数
    save_steps=config["training"]["save_steps"],  # 保存间隔
    save_total_limit=config["training"]["save_total_limit"],  # 最多保存的模型数量
    max_grad_norm=config["training"]["max_grad_norm"],  # 梯度裁剪阈值
    dataloader_pin_memory=config["training"]["dataloader_pin_memory"],  # 是否对数据加载器中的数据进行内存锁定
    dataloader_num_workers=config["training"]["dataloader_num_workers"],
    seed=RANDOM_SEED,  # 使用已定义的RANDOM_SEED变量
    report_to=config["training"]["report_to"],
    run_name=config["training"]["run_name"].format(LORA_R=LORA_R, target_modules_layer=target_modules_layer[-1]),
    output_dir=config["training"]["output_dir"].format(model_name=model_name, LORA_R=LORA_R,
                                                       target_modules_layer=target_modules_layer[-1]),
)


# ========== 模型 ==========
def build_model_unsloth():
    if USE_LORA:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            # _attn_implementation="flash_attention_2"
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID)

        if config["from_lora_checkpoint"] is not None:
            print("合并原来的lora")
            model = PeftModel.from_pretrained(model, config["from_lora_checkpoint"])
            model = model.merge_and_unload()
        # ----------------------------------------------------------------------------------
        # lora config
        target_modules = ([f'model.text_model.layers.{i}.self_attn.q_proj' for i in target_modules_layer] +
                          [f'model.text_model.layers.{i}.self_attn.k_proj' for i in target_modules_layer] +
                          [f'model.text_model.layers.{i}.self_attn.v_proj' for i in target_modules_layer] +
                          [f'model.text_model.layers.{i}.self_attn.o_proj' for i in target_modules_layer] +
                          [f'model.text_model.layers.{i}.mlp.gate_proj' for i in target_modules_layer] +
                          [f'model.text_model.layers.{i}.mlp.up_proj' for i in target_modules_layer] +
                          [f'model.text_model.layers.{i}.mlp.down_proj' for i in target_modules_layer])
        target_modules.append("model.connector.modality_projection.proj")
        target_modules.append("lm_head")

        # 创建一个字典，为每个模块指定rank值
        rank_pattern = {}
        for layer_idx in target_modules_layer:
            layer_rank = LORA_RANKS.get(layer_idx, LORA_R)
            for module_type in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                                'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
                module_name = f'model.text_model.layers.{layer_idx}.{module_type}'
                rank_pattern[module_name] = layer_rank

        # 为lm_head设置默认rank
        rank_pattern["lm_head"] = LM_HEAD_RANk
        rank_pattern["model.connector.modality_projection.proj"] = CONNECT_RANK

        lora_config = LoraConfig(
            r=LORA_R,  # 默认rank值
            lora_alpha=LORA_R,  # 默认alpha值
            lora_dropout=0.1,
            target_modules=target_modules,
            use_dora=True,
            init_lora_weights="gaussian",
            rank_pattern=rank_pattern,  # 使用rank_pattern为不同模块指定不同的rank
            alpha_pattern=rank_pattern
        )
        lora_config.inference_mode = False

        # -------------------------------------------------------------------------------
        model.add_adapter(lora_config)
        model.enable_adapters()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model = model.to("cuda", dtype=torch.bfloat16 if bf_16 else torch.float16)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            # _attn_implementation="flash_attention_2"
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID)

        if config["from_lora_checkpoint"] is not None:
            model = PeftModel.from_pretrained(model, config["from_lora_checkpoint"])
            model = model.merge_and_unload()

        for param in model.model.connector.parameters():
            param.requires_grad = False
        for layer in model.model.text_model.layers:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in model.model.text_model.layers[-5:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True
        # 如果只想微调LLM部分，冻结视觉模型参数
        for param in model.model.vision_model.parameters():
            param.requires_grad = False
    return processor, model


def load_processor_and_model():
    if __name__ == "__main__":
        return build_model_unsloth()
    else:
        return AutoProcessor.from_pretrained(MODEL_ID), None


def print_model_info(model):
    print(model.model)
    peak_mem = torch.cuda.max_memory_allocated()
    print(f"模型当前占用显存: {peak_mem / 1024 ** 3:.2f} GB")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable_params / 1e5:.2f} 万")
    print(f"总参数: {total_params / 1e5:.2f} 万")


def collate_fn(examples):
    instances = []
    prompts = []
    answers = []
    images = []
    data_type = []
    for example in examples:
        image = os.path.join(image_dir, example["image"].replace("\\", "/"))
        question = example["conversations"][0]["content"].replace("\n<image>", "")
        # 去除问题中的字数限制提示
        if question.startswith("Write"):
            question = question.replace("100 words ", "")
        elif question.startswith("Analyze"):
            question = question.replace(" Keep it 100 words.", "")
        answer = example["conversations"][1]["content"]

        user_content = [{"type": "text", "text": question},
                        {"type": "image", "path": image}]

        messages = [
            {"role": "user", "content": user_content},
        ]

        instance = processor.apply_chat_template(messages, add_generation_prompt=True,
                                                 tokenize=True, return_dict=True,
                                                 return_tensors="pt").to("cuda").to(dtype)
        instances.append(instance)
        prompts.append(question)
        answers.append(answer)
        images.append(image)
        data_type.append(example.get("type", "desc"))

    input_ids = pad_sequence(
        [inst["input_ids"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id
    )
    attention_mask = pad_sequence(
        [inst["attention_mask"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=0
    )
    labels = pad_sequence(
        [inst["input_ids"].squeeze(0).clone() for inst in instances],
        batch_first=True,
        padding_value=-100
    )

    labels[labels == image_token_id] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompts": prompts,
        "answers": answers,
        "images": images,
        "data_type": data_type,
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
    for ex in instances:
        pv = ex.get("pixel_values", None).squeeze(0)

        if pv is None:
            # text-only => fill pixel data + mask with zeros
            shape_pv = (max_frames, 3, max_h, max_w)
            padded_pv = torch.zeros(shape_pv, dtype=dtype)
        else:
            f, c, h, w = pv.shape
            # Prepare final storage
            padded_pv = torch.zeros(
                (max_frames, c, max_h, max_w),
                dtype=pv.dtype,
                device=pv.device
            )
            padded_pv[:f, :, :h, :w] = pv
        padded_pixel_values_list.append(padded_pv)

    out["pixel_values"] = torch.stack(padded_pixel_values_list, dim=0)
    return out


def load_dataset_fn():
    print("加载数据集...")
    data_files = {"train": dataset_path}
    if test_dataset_path is not None:
        data_files["test"] = test_dataset_path
    raw_dataset = load_dataset("json", data_files=data_files)
    train_ds = raw_dataset["train"]
    if test_dataset_path is not None:
        test_ds = raw_dataset["test"]
    else:
        test_ds = None
    print("数据集加载完成！")
    print("训练集样例：", train_ds[0])
    # print("测试集样例：", test_ds[0])
    return train_ds, test_ds


# 获取图像token_id
def get_image_token_id(processor):
    return processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")]


processor, model = load_processor_and_model()
image_token_id = get_image_token_id(processor)
dtype = torch.bfloat16 if bf_16 or model is None else model.dtype

# ---------------------------------------------------------------------------
import json
import requests
from PIL import Image
from io import BytesIO
import base64


def image_to_base64(input_path, output_quality=None):
    try:
        # 读取图片
        with Image.open(input_path) as img:
            # 转换为RGB模式（JPEG不支持RGBA）
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            # 创建内存缓冲区
            buffer = BytesIO()

            if img.size[0] > 1920 or img.size[1] > 1920:
                # 按照比例缩放图片
                scale = max(img.size[0] / 1920, img.size[1] / 1920)
                new_size = (int(img.size[0] / scale), int(img.size[1] / scale))
                img = img.resize(new_size, Image.LANCZOS)

            img.save(buffer, format='PNG', quality=90)

            # 获取二进制数据
            img_bytes = buffer.getvalue()

            # 转换为Base64编码
            encoded = base64.b64encode(img_bytes).decode('utf-8')

            return encoded
    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_path}'")
        return None
    except Exception as e:
        print(f"错误：处理图片时发生异常：{e}")
        return None


# 保存mimo reward产生的数据
def save_mimo_reward_data(image_path, question, vlm_answer, score_prompt, response_json, data_type):
    data = {
        "image_path": image_path,
        "question": question,
        "vlm_answer": vlm_answer,
        "score_prompt": score_prompt,
        "response_json": response_json,
        "data_type": data_type
    }
    with open(config["mimo_reward_data_save_path"], "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def score_with_mino_desc(image_path, question, desc, data_type):
    api_url = config["llama_server"]["api_url"] + "v1/chat/completions"
    prompt_template = "下面是对这张图片的一个描述，请逐句分析描述是否正确并记录下来，最后使用json格式返回正确句子的数量与错误句子的数量。\njson格式：{{'correct_count': 正确句子数量, 'wrong_count': 错误句子数量}}\n描述：{desc}"
    prompt_template = prompt_template.format(desc=desc)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(image_path)}"}},
                    {"type": "text", "text": prompt_template},
                ]
            },
        ],
        "temperature": 0.1,
        "max_tokens": 8000,
        "model": "mimo-vl-7b-rl",
    }
    try_time = 0
    while True:
        try_time += 1
        if try_time > 3:
            print("!!![score_with_mino_desc]尝试请求3次失败")
            return 0
        response = requests.post(api_url, json=payload, timeout=600)
        if response.status_code == 200:
            try:
                data = response.json()
                content = data['choices'][0]['message']['content']
                if len(content.split("<think>")[1].split("</think>")[0])<20:
                    time.sleep(0.05)
                    continue
                score_str = content.split("</think>")[1]
                if "```json" in score_str:
                    score_str = score_str.replace("```json", "").replace("```", "")
                score_data = json.loads(score_str)
                correct_count = float(score_data['correct_count'])
                wrong_count = float(score_data['wrong_count'])
                # 保存评分结果
                save_mimo_reward_data(image_path, question, desc, prompt_template, data, data_type)
                return ((correct_count - 1.1 * wrong_count) / (correct_count + wrong_count) + 1.1) / 2.2  # 归一化
            except Exception as e:
                print(f"[score_with_mino_desc] Error: {e}")
                print(f"response: ", response.json())
                time.sleep(0.05)
                continue
        else:
            print(f"[score_with_mino_desc]Error: {response.status_code}")
            time.sleep(0.05)
            continue


def score_with_mino_QA(image_path, question, answer, data_type):
    api_url = config["llama_server"]["api_url"] + "v1/chat/completions"
    prompt_template = f"下面是关于这张图片的一个问答，请分析回答是否正确，在回答时只输出‘正确’或‘错误’\n问题：{question}\n回答：{answer}"

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(image_path)}"}},
                    {"type": "text", "text": prompt_template},
                ]
            },
        ],
        "temperature": 0.1,
        "max_tokens": 8000,
        "model": "mimo-vl-7b-rl",
    }
    try_time = 0
    while True:
        try_time += 1
        if try_time > 3:
            print("!!![score_with_mino_QA]尝试请求3次失败")
            return 0
        response = requests.post(api_url, json=payload, timeout=600)
        if response.status_code == 200:
            try:
                data = response.json()
                content = data['choices'][0]['message']['content']
                if len(content.split("<think>")[1].split("</think>")[0])<20:
                    time.sleep(0.05)
                    continue
                score_str = content.split("</think>")[1]
                # 保存评分结果
                save_mimo_reward_data(image_path, question, answer, prompt_template, data, data_type)
                if "正确" in score_str and "不正确" not in score_str:
                    return 1.0
                else:
                    return 0.0
            except Exception as e:
                print(f"[score_with_mino_QA]Error: {e}")
                time.sleep(0.05)
                continue
        else:
            print(f"[score_with_mino_QA]Error: {response.status_code}")
            time.sleep(0.05)
            continue


# 奖励函数
def mimo_reward(prompts, completions, **reward_kwargs):
    """
    奖励函数
    :param prompts: 样本
    :param completions: 模型输出
    :param reward_kwargs: 其他参数
    :return: 奖励
    """
    reward = []
    for img_p, completion, prompt, data_type in zip(reward_kwargs["images"],
                                                    completions, prompts,
                                                    reward_kwargs["data_type"]):
        if check_repeat(completion):
            reward.append(0.0)
        else:
            if len(completion) <= 30:
                reward.append(0.0)
            else:
                if data_type == "desc":
                    score = score_with_mino_desc(img_p, prompt, completion, data_type)
                else:
                    score = score_with_mino_QA(img_p, prompt, completion, data_type)
            # score = 0.0
            time.sleep(0.05)
            reward.append(score)

    return reward


# 检查是否有重复内容,若有则返回True
def check_repeat(text: str) -> bool:
    # 去除空白符
    # text = content.replace('\n', '').replace(' ', '')
    # 设定最小重复片段长度（比如15个字），和最大允许重复次数
    max_len = 40
    max_repeat = 2
    n = len(text)
    # 只检测较长文本
    if n < max_len * 2:
        return False
    # 用滑动窗口检测重复片段
    res = False
    for size in range(max_len, max_len - 15, -1):
        substr_count = {}
        for i in range(n - size + 1):
            substr = text[i:i + size]
            if substr in substr_count:
                substr_count[substr] += 1
            else:
                substr_count[substr] = 1
            if substr_count[substr] >= max_repeat:
                return True
    return False


def repeat_penalty(prompts, completions, **reward_kwargs):
    """
    重复惩罚
    :param prompts: 样本
    :param completions: 模型输出
    :param reward_kwargs: 其他参数
    :return: 惩罚
    """
    penalty = []
    for completion in completions:
        if check_repeat(completion):
            penalty.append(-1)
        else:
            penalty.append(0)

    return penalty


def total_len_reward(prompts, completions, **reward_kwargs):
    """
    长度奖励函数
    :param prompts: 样本
    :param completions: 模型输出
    :param reward_kwargs: 其他参数
    :return: 奖励
    """

    reward = []
    for completion, data_type in zip(completions, reward_kwargs["data_type"]):
        if data_type == "desc":
            len_completion = len(completion)
            if len_completion <= 30:
                reward.append(-1.0)
            elif len_completion <= 100 or check_repeat(completion):
                reward.append(0.0)
            elif len_completion <= 200:
                reward.append(0.1 + 0.1 * (200 - len_completion) / 100)
            else:
                reward.append(0.2)
        else:
            reward.append(0.0)
    return reward


# ---------------------------------------------------------------------------


from trl.trainer.grpo_trainer import *


class VLMGRPOTrainer(GRPOTrainer):
    def __init__(self,
                 model,
                 reward_funcs,
                 args: GRPOConfig = None,
                 train_dataset=None,
                 eval_dataset=None,
                 processing_class=None,
                 reward_processing_classes=None,
                 callbacks=None,
                 optimizers=(None, None),
                 peft_config=None,
                 data_collator=None
                 ):
        super().__init__(model=model,
                         reward_funcs=reward_funcs,
                         args=args,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         processing_class=processing_class,
                         reward_processing_classes=reward_processing_classes,
                         callbacks=callbacks,
                         # optimizers=optimizers,
                         peft_config=peft_config)

        self.data_collator = data_collator

    def _prepare_inputs(self, inputs):
        device = self.accelerator.device
        prompts = inputs["prompts"]
        # prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        # prompt_inputs = self.processing_class(
        #     prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        # )
        prompt_inputs = Trainer._prepare_inputs(self, inputs)
        prompt_ids = prompt_inputs["input_ids"].to(device=self.model.device)
        prompt_mask = prompt_inputs["attention_mask"].to(device=self.model.device)
        pixel_values = inputs["pixel_values"].to(device=self.model.device)

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Regular generation path
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                input_ids=prompt_ids, pixel_values=pixel_values, attention_mask=prompt_mask,
                generation_config=self.generation_config
            )

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # if is_conversational(inputs[0]):
        #     completions = []
        #     for prompt, completion in zip(prompts, completions_text):
        #         bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
        #         completions.append([{"role": "assistant", "content": bootstrap + completion}])
        # else:
        #     completions = completions_text
        completions = completions_text  # 跳过上面的判断
        # print("completions:" + str(completions))

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                # if is_conversational(inputs[0]):
                #     messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                #     texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                # else:
                # texts = [p + c for p, c in zip(prompts, completions)]
                texts = [p + c for p, c in zip(prompts, completions)]  # 跳过上面的判断

                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                # keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                # reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                keys = [key for key in inputs if key not in ["prompts", "completions"]]
                reward_kwargs = {key: inputs[key] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        if (
                self.log_completions
                and self.state.global_step % self.args.logging_steps == 0
                and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }


def train_main():
    processor, model = load_processor_and_model()  # 加载处理器和模型
    print_model_info(model)  # 打印模型信息
    train_ds, test_ds = load_dataset_fn()  # 加载数据集

    processor.pad_token_id = processor.tokenizer.eos_token_id

    trainer = VLMGRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[
            mimo_reward,
            repeat_penalty,
            total_len_reward
        ],
        args=training_args,
        train_dataset=train_ds,
        data_collator=collate_fn,
    )

    print("开始训练...")
    trainer.train()
    print("训练完成！")
    # eval_results = trainer.evaluate()
    # print("评估结果：", eval_results)


if __name__ == "__main__":
    train_main()

