from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel, LoraConfig  # 导入 LoraConfig
import torch
import os
import time
import json  # 导入 json
import pandas as pd
from transformers import TextStreamer
import yaml  # 导入yaml库
import subprocess
import requests
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
import asyncio
from tqdm import tqdm


# 加载配置文件
def load_config(config_path="eval_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 处理模板字符串
    if "{lora_step}" in config["model"]["lora_adapter_path"]:
        config["model"]["lora_adapter_path"] = config["model"]["lora_adapter_path"].format(
            lora_step=config["model"]["lora_step"])

    if "{image_dir}" in config["eval"]["excel_path"] or "{lora_step}" in config["eval"]["excel_path"]:
        config["eval"]["excel_path"] = config["eval"]["excel_path"].format(
            image_dir=config["eval"]["image_dir"],
            lora_step=config["model"]["lora_step"])

    return config


# 加载配置
config = load_config()

# 设置环境变量
os.environ["HF_ENDPOINT"] = config["env"]["hf_endpoint"]
os.environ["ARK_API_KEY"] = config["env"]["ark_api_key"]

# 停止运行模型
subprocess.run(['ollama', 'list'])
subprocess.run(['ollama', 'stop', config["translation"]["model"]])

# 基础模型的路径或Hugging Face模型ID
base_model_path = config["model"]["base_model_path_small"] if config["model"]["smol"] else config["model"][
    "base_model_path_large"]
lora_step = config["model"]["lora_step"]
skip_generate_and_translate = config["eval"]["skip_generate_and_translate"]
skip_score = config["eval"]["skip_score"]

# 训练好的LoRA适配器路径
lora_adapter_path = config["model"]["lora_adapter_path"]

# 测试图像的目录
image_dir = config["eval"]["image_dir"]

# Excel 文件路径
excel_path = config["eval"]["excel_path"]

if not skip_generate_and_translate:
    # 加载处理器
    processor = AutoProcessor.from_pretrained(base_model_path)

    # 加载基础模型
    base_model_for_lora = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2"
    )

    if lora_step > 0:
        print("加载LORA...")
        if config["model"]["base_adapter_path"] is not None:
            # 加载LoRA适配器到基础模型，并传入修正后的config
            model = PeftModel.from_pretrained(base_model_for_lora, config["model"]["base_adapter_path"])
            model = model.merge_and_unload()
        else:
            model = base_model_for_lora

        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model = model.merge_and_unload().eval()

        model = model.to("cuda").to(torch.bfloat16)
    else:
        print("使用基础模型...")
        model = base_model_for_lora
        model = model.to("cuda").to(torch.bfloat16).eval()

    peak_mem = torch.cuda.max_memory_allocated()
    print(f"模型当前占用显存: {peak_mem / 1024 ** 3:.2f} GB")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数: {total_params / 1e5:.2f} 万")
else:
    model = None
    processor = None

if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
    print(f"读取到excel文件 共{len(df)}行")
    # 添加一行分隔
    df = pd.concat([df, pd.DataFrame(
        [{"image_file": "一一一",
          "output_str": "一一一",
          "output_str_beam": "一一一",
          "score": "一一一",
          "score_beam": "一一一",
          "reasoning": "一一一",
          "reasoning_beam": "一一一",
          "translate": "一一一",
          "translate_beam": "一一一"}])],
                   ignore_index=True)
else:
    df = pd.DataFrame(columns=["image_file",
                               "output_str",
                               "output_str_beam",
                               "score",
                               "score_beam",
                               "reasoning",
                               "reasoning_beam",
                               "translate",
                               "translate_beam"])


def caption_one_image(image_path):
    with torch.no_grad():
        prompts = ["Write a descriptive caption for this image in a formal tone.",
                   "Write a descriptive caption for this image in a casual tone.",
                   "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. ",
                   ]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text",
                     "text": prompts[0]},
                ]
            },
        ]
        st = time.time()

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        print(f"处理输入耗时: {time.time() - st:.4f} s")

        print(f'[Image]: {os.path.basename(image_path)}')
        print('🤖️: ', end='')
        # streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # 使用配置中的生成参数
        gen_config = config["generation"]
        generated_ids = model.generate(
            **inputs,
            do_sample=gen_config["do_sample"],
            temperature=gen_config["temperature"],
            top_p=gen_config["top_p"],
            max_length=gen_config["max_length"],
            no_repeat_ngram_size=gen_config["no_repeat_ngram_size"],
            repetition_penalty=gen_config["repetition_penalty"],
        )

        beam_config = gen_config["beam_generation"]
        generated_ids2 = model.generate(
            **inputs,
            num_beams=beam_config["num_beams"],
            max_length=beam_config["max_length"],
            early_stopping=beam_config["early_stopping"],
            no_repeat_ngram_size=beam_config["no_repeat_ngram_size"],
            length_penalty=beam_config["length_penalty"],
            repetition_penalty=beam_config["repetition_penalty"],
        )
        # 如果还需要完整文本，可以再decode一次
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        generated_texts.extend(processor.batch_decode(
            generated_ids2,
            skip_special_tokens=True,
        ))
        output_str = generated_texts[0].split("Assistant:")[-1].strip()
        output_str2 = generated_texts[1].split("Assistant:")[-1].strip()
        print(output_str)
        return st, output_str, output_str2, generated_ids, generated_ids2


def caption_dir(image_dir):
    if not os.path.exists(image_dir):
        return
    global df
    global_token_speed = [0, 0]
    global_chart_speed = [0, 0]
    # 遍历图像文件进行测试
    for image_file in os.listdir(image_dir):
        if image_file.endswith(".xlsx"):  # 跳过非图像文件
            continue

        image_path = os.path.join(image_dir, image_file)
        if os.path.isdir(image_path):
            continue

        st, output_str, output_str2, generated_ids, generated_ids2 = caption_one_image(image_path)

        et = time.time() - st
        token_len = len(generated_ids[0]) + len(generated_ids2[0])
        chart_len = len(output_str) + len(output_str2)
        token_speed = token_len / et
        chart_speed = chart_len / et
        global_token_speed[0] += token_len
        global_token_speed[1] += et
        global_chart_speed[0] += chart_len
        global_chart_speed[1] += et
        print(f"token_speed:{token_speed:.4f} tokens/s")
        print(f"chart_speed:{chart_speed:.4f} chart/s")
        print(f"总耗时:{et:.4f} s")
        print("-" * 50)

        # 新增：将输出写入Excel文件
        new_row = {
            "image_file": image_path,
            "output_str": output_str,
            "output_str_beam": output_str2,
            "score": None,
            "score_beam": None,
            "reasoning": None,
            "reasoning_beam": None,
            "translate": None,
            "translate_beam": None
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    print(f"global_token_speed:{global_token_speed[0] / global_token_speed[1]:.4f} tokens/s")
    print(f"global_chart_speed:{global_chart_speed[0] / global_chart_speed[1]:.4f} chart/s")


if skip_generate_and_translate:
    safe_len = len([i for i in os.listdir(image_dir) if not i.endswith(".xlsx")])
else:
    caption_dir(image_dir)
    safe_len = df.shape[0]
    caption_dir(os.path.join(image_dir, "R18"))

# 卸载模型，释放显存
del model
torch.cuda.empty_cache()

# 初始化OpenAI客户端
client = OpenAI(
    base_url=config["scoring"]["base_url"],
    api_key=os.environ.get("ARK_API_KEY"),
)


def translate_with_ollama(text):
    # 使用chat模式，构造翻译提示词
    url = "http://localhost:11434/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    messages = [
        {"role": "system",
         "content": config["translation"]["system_prompt"]},
        {"role": "user", "content": text}
    ]
    payload = {
        "model": config["translation"]["model"],
        "messages": messages,
        "max_tokens": config["translation"]["max_tokens"],
        "temperature": config["translation"]["temperature"]
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=240)
        if response.status_code == 200:
            data = response.json()
            # 兼容不同返回格式
            if "choices" in data and len(data["choices"]) > 0:
                res = data["choices"][0]["message"]["content"].strip()
                if "</think>" in res:
                    return res.split("</think>")[1].strip()
                else:
                    return res
            else:
                return ""
        else:
            print(f"翻译请求失败，状态码: {response.status_code}")
            return ""
    except Exception as e:
        print(f"翻译失败: {e}")
        return ""


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


def score_with_doubao(text, image_path):
    # 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
    # 初始化Ark客户端，从环境变量中读取您的API Key
    prompt = config["scoring"]["prompt"].format(text=text)

    response = client.chat.completions.create(
        model=config["scoring"]["model"],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_to_base64(image_path)}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    reasoning = response.choices[0].message.reasoning_content
    answer = response.choices[0].message.content
    print(reasoning + "\n" + answer)
    if "<score>" in answer:
        answer = answer.split("<score>")[-1].split("</score>")[0].strip()
    return reasoning, answer


async def async_translate_with_ollama(output_str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, translate_with_ollama, output_str)


async def async_score_with_doubao(text, image_path):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, score_with_doubao, text, image_path)


async def async_main(skip_translate=False):
    global df
    if not skip_translate:
        # 创建一个与DataFrame长度相同的空列表
        translations = [""] * len(df)
        translations_beam = [""] * len(df)

        # 创建任务列表，并保持与DataFrame索引的对应关系
        tasks = []
        for idx, output_str in enumerate(df["output_str"]):
            if output_str == "一一一":
                continue
            # 跳过已经翻译过的行
            if not pd.isna(df.at[idx, "translate"]):
                translations[idx] = df.at[idx, "translate"]
                continue
            # 将索引与任务一起存储，以便后续匹配
            task = asyncio.create_task(async_translate_with_ollama(output_str))
            tasks.append((idx, "output_str", task))

        for idx, output_str_beam in enumerate(df["output_str_beam"]):
            if output_str_beam == "一一一":
                continue
            # 跳过已经翻译过的行
            if not pd.isna(df.at[idx, "translate_beam"]):
                translations_beam[idx] = df.at[idx, "translate_beam"]
                continue
            # 将索引与任务一起存储，以便后续匹配
            task = asyncio.create_task(async_translate_with_ollama(output_str_beam))
            tasks.append((idx, "output_str_beam", task))

        # 使用tqdm显示进度
        for idx, col, task in tqdm([(i, c, t) for i, c, t in tasks], total=len(tasks), desc="Translating"):
            result = await task
            # 将翻译结果放入对应的位置
            if col == "output_str_beam":
                translations_beam[idx] = result
            else:
                translations[idx] = result

        # 将翻译结果添加到DataFrame
        df["translate"] = translations
        df["translate_beam"] = translations_beam

        # 保存到 Excel
        df.to_excel(excel_path, index=False)
        print("完成翻译, 已保存")
        print("开始评分...")

    # =============================进行评分================================
    if skip_score:
        return
    scores = [""] * len(df)
    scores_beam = [""] * len(df)
    reasoning_col = [""] * len(df)
    reasoning_col_beam = [""] * len(df)

    # 添加信号量控制并发
    sem = asyncio.Semaphore(config["scoring"]["concurrent_tasks"])  # 可根据需要调整并发数

    # 定义带信号量限制的任务包装函数
    async def limited_task(coro):
        async with sem:  # 同一时间最多运行sem.value个任务
            return await coro

    # 创建任务列表，并保持与DataFrame索引的对应关系
    tasks = []
    for idx in range(safe_len):
        if df.at[idx, "output_str"] == "一一一":
            continue
        # 跳过已经翻译过的行
        if not pd.isna(df.at[idx, "score"]):
            scores[idx] = df.at[idx, "score"]
            continue
        # 将索引与任务一起存储，以便后续匹配
        image_path = df.at[idx, "image_file"]
        output_str = df.at[idx, "output_str"]
        # 包装任务以应用信号量限制
        task = asyncio.create_task(limited_task(async_score_with_doubao(output_str, image_path)))
        tasks.append((idx, "output_str", task))

    for idx in range(safe_len):
        if df.at[idx, "output_str_beam"] == "一一一":
            continue
        # 跳过已经翻译过的行
        if not pd.isna(df.at[idx, "score_beam"]):
            scores_beam[idx] = df.at[idx, "score_beam"]
            continue
        # 将索引与任务一起存储，以便后续匹配
        image_path = df.at[idx, "image_file"]
        output_str_beam = df.at[idx, "output_str_beam"]
        # 包装任务以应用信号量限制
        task = asyncio.create_task(limited_task(async_score_with_doubao(output_str_beam, image_path)))
        tasks.append((idx, "output_str_beam", task))

    # 使用tqdm显示进度
    for idx, col, task in tqdm([(i, c, t) for i, c, t in tasks], total=len(tasks), desc="Scoring"):
        reasoning, answer = await task
        # 将翻译结果放入对应的位置
        if col == "output_str_beam":
            scores_beam[idx] = answer
            reasoning_col_beam[idx] = reasoning
        else:
            scores[idx] = answer
            reasoning_col[idx] = reasoning

    # 将评分结果添加到DataFrame
    df["score"] = scores
    df["score_beam"] = scores_beam
    df["reasoning"] = reasoning_col
    df["reasoning_beam"] = reasoning_col_beam


asyncio.run(async_main(skip_generate_and_translate))

# 保存到 Excel
df.to_excel(excel_path, index=False)

print("Evaluation script finished.")
subprocess.run(['ollama', 'stop', config["translation"]["model"]])
