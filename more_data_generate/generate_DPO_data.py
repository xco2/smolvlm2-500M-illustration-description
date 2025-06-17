import asyncio
import signal
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import aiohttp
from PIL import Image
from io import BytesIO
import base64
import os
import tqdm
import json
import re
import yaml
import random

# 定义一个标志变量，用于指示是否收到停止信号
stop_requested = False


def signal_handler(sig, frame):
    global stop_requested
    stop_requested = True
    print("收到停止信号，将在保存完当前数据后退出...")


# 注册信号处理函数
signal.signal(signal.SIGINT, signal_handler)


def image_to_base64(input_path, output_quality=None):
    try:
        # 读取图片
        with Image.open(input_path) as img:
            # 转换为RGB模式（JPEG不支持RGBA）
            if img.mode == "RGBA":
                img = img.convert("RGB")
            # 创建内存缓冲区
            buffer = BytesIO()

            if img.size[0] > 1920 or img.size[1] > 1920:
                # 按照比例缩放图片
                scale = max(img.size[0] / 1920, img.size[1] / 1920)
                new_size = (int(img.size[0] / scale), int(img.size[1] / scale))
                img = img.resize(new_size, Image.LANCZOS)

            img.save(buffer, format="PNG", quality=90)

            # 获取二进制数据
            img_bytes = buffer.getvalue()

            # 转换为Base64编码
            encoded = base64.b64encode(img_bytes).decode("utf-8")

            return encoded
    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_path}'")
        return None
    except Exception as e:
        print(f"错误：处理图片时发生异常：{e}")
        return None


# 加载配置文件
def load_config(config_path="generate_multi_results_config.yaml"):
    """加载YAML配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 读取jsonl文件
def load_jsonl(file_path):
    """加载jsonl文件并返回数据列表"""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    except Exception as e:
        print(f"加载数据失败: {e}")
        return []


# 保存jsonl文件
def save_jsonl(data, file_path):
    """保存数据列表到jsonl文件"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"保存数据失败: {e}")
        return False


# 请求大模型
async def async_chat_with_llm(
        image_path,
        prompt,
        api_url,
        model_name="mimo-vl-7b-rl",
        temperature=0.1,
        max_tokens=8000,
):
    """
    请求大模型
    """
    # 准备请求负载
    payload = {
        "messages": [
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
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "model": model_name,
    }

    # 尝试最多3次请求
    for try_time in range(1, 4):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, timeout=600) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        return content
        except Exception as e:
            print(f"[async_score_with_mino_desc] 尝试 {try_time}/3 失败: {e}")
            if try_time == 3:
                print("!!![async_score_with_mino_desc]尝试请求3次失败!!!")
                return None


# 检测是否有多个重复短语
def has_many_short_words_sequence(
        text, words_threshold=2, min_phrases=5, min_total_words=20
):
    """
    判断文本中是否包含大量逗号分隔的短词排列

    参数:
    text (str): 输入的英文字符串
    words_threshold (int): 词数少于words_threshold的部分视为短句
    min_phrases (int): 短句大于min_phrases被视为有很多短句
    min_total_words (int): 文本中的最小总单词数

    返回:
    bool: 如果包含很多连续短词排列则返回True，否则返回False
    """
    # 计算文本中的总单词数
    total_words = len(re.findall(r"\b\w+\b", text))

    if total_words < min_total_words:
        return False

    # 分割文本为短语（以逗号分隔）
    phrases = re.split(r"[,.]\s*", text)

    # 计算符合条件的短语数量（短词短语）
    consecutive_short_phrases = 0  # 连续短词短语计数
    max_consecutive_found = 0  # 记录找到的最大连续短词数量

    for phrase in phrases:
        # 跳过空短语
        if not phrase.strip():
            consecutive_short_phrases = 0  # 重置连续计数
            continue

        # 计算短语中的单词数
        words_in_phrase = len(re.findall(r"\b\w+\b", phrase))

        # 如果短语中的单词数小于等于阈值，认为是短词短语
        if words_in_phrase <= words_threshold:
            consecutive_short_phrases += 1
            max_consecutive_found = max(
                max_consecutive_found, consecutive_short_phrases
            )

            # 检查是否满足连续短词和总数的双重条件
            if max_consecutive_found >= min_phrases:
                return True
        else:
            consecutive_short_phrases = 0  # 重置连续计数

    # 判断是否同时满足总短词数量和连续短词数量的要求
    return max_consecutive_found >= min_phrases


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
            substr = text[i: i + size]
            if substr in substr_count:
                substr_count[substr] += 1
            else:
                substr_count[substr] = 1
            if substr_count[substr] >= max_repeat:
                return True
    return False


class DataProcessor(ABC):
    """数据处理基类，抽象了数据遍历、请求大模型和保存数据的流程"""

    def __init__(self, config: Dict[str, Any], step_name: str):
        self.config = config
        self.results = []
        self.step_name = step_name

    def load_data(self, jsonl_file_path: str) -> List[Dict[str, Any]]:
        """加载数据，子类可以重写此方法来支持不同的数据格式"""
        return load_jsonl(jsonl_file_path)

    @abstractmethod
    def preprocess_item(
            self, item: Dict[str, Any], image_dir: str
    ) -> Optional[Dict[str, Any]]:
        """预处理单个数据项，返回处理后的数据或None（跳过）

        Args:
            item: 原始数据项
            image_dir: 图片目录

        Returns:
            处理后的数据字典，包含必要的字段，或None表示跳过
        """
        pass

    async def request_model_with_id(
            self, processed_item: Dict[str, Any]
    ) -> tuple[int, Optional[str]]:
        model_response = await self.request_model(processed_item)
        return processed_item["id"], model_response

    @abstractmethod
    async def request_model(self, processed_item: Dict[str, Any]) -> Optional[str]:
        """请求大模型，子类需要实现具体的请求逻辑

        Args:
            processed_item: 预处理后的数据项

        Returns:
            模型响应内容或None
        """
        pass

    @abstractmethod
    def postprocess_result(
            self, processed_item: Dict[str, Any], model_response: List[str]
    ) -> Optional[Dict[str, Any]]:
        """后处理结果，子类可以实现自定义的结果格式

        Args:
            processed_item: 预处理后的数据项
            model_response: 模型响应

        Returns:
            最终结果字典或None（跳过）
        """
        pass

    def tag_origin_answer_data(self, item: Dict[str, Any]) -> dict:
        """为原始数据打标签，子类可以重写此方法实现自定义过滤逻辑

        Args:
            item: 原始数据项

        Returns:
            标签
        """
        return {}

    def tag_output_data(self, result_item: Dict[str, Any]) -> dict:
        """为输出数据打标签，子类可以重写此方法实现自定义过滤逻辑

        Args:
            result_item: 处理后的结果项

        Returns:
            标签
        """
        return {}

    async def process_batch(
            self, batch_items: List[Dict[str, Any]], image_dir: str
    ) -> List[Dict[str, Any]]:
        """处理一个批次的数据"""
        tasks = []
        item_map = {}

        # 预处理并创建任务
        for idx, item in enumerate(batch_items):
            # 预处理模型输入数据
            processed_item = self.preprocess_item(item, image_dir)
            if processed_item is None:
                continue

            processed_item.update({"id": idx})

            # 创建异步任务, 循环构造任务生成多个
            for _ in range(self.config[self.step_name]["num_genreate"]):
                task = self.request_model_with_id(processed_item)
                tasks.append(task)

            # 保存任务索引与数据的映射关系
            item_map[idx] = {
                "original_item": item,
                "processed_item": processed_item,
            }

        # 并发执行所有任务
        if not tasks:
            return []

        responses = await asyncio.gather(*tasks)

        # 合并多个输出
        merged_responses = {}
        for idx, response in responses:
            if response is not None:
                if idx not in merged_responses:
                    merged_responses[idx] = []
                merged_responses[idx].append(response)

        # 处理结果
        batch_results = []
        for idx, response in merged_responses.items():
            item_data = item_map[idx]
            result_item = self.postprocess_result(item_data["processed_item"], response)

            if result_item is not None:
                batch_results.append(result_item)

        return batch_results

    def save_results(self, output_path: str) -> None:
        """保存结果，子类可以重写此方法实现不同的保存格式"""
        with open(output_path, "w", encoding="utf-8") as f:
            for item in self.results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def save_progress(self, current_index: int, output_path: str) -> None:
        """保存处理进度"""
        progress_file = output_path.replace(".jsonl", "_progress.json")
        progress_data = {
            "current_index": current_index,
            "total_results": len(self.results),
            "output_path": output_path,
        }
        print(f"进度已保存到 {progress_file}")
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)

    def load_progress(self, output_path: str) -> int:
        """加载处理进度，返回应该开始的索引"""
        progress_file = output_path.replace(".jsonl", "_progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, "r", encoding="utf-8") as f:
                    progress_data = json.load(f)

                # 如果输出文件存在，加载已有结果
                if os.path.exists(output_path):
                    print(
                        f"发现进度文件，从索引 {progress_data['current_index']} 继续处理..."
                    )
                    self.results = self.load_data(output_path)
                    print(f"已加载 {len(self.results)} 条已处理的数据")
                    return progress_data["current_index"]
            except Exception as e:
                print(f"加载进度文件失败: {e}，从头开始处理")
        return 0

    def process_data(
            self, jsonl_file_path: str, image_dir: str
    ) -> List[Dict[str, Any]]:
        """主处理流程"""
        print("加载数据...")
        data = self.load_data(jsonl_file_path)
        # 打乱数据
        random.seed(self.config["random_seed"])
        random.shuffle(data)
        data_len = len(data)
        print(f"共加载 {data_len} 条数据")

        # 获取输出路径
        output_path = self.config[self.step_name]["output_path"]

        # 加载进度，获取开始索引
        start_index = self.load_progress(output_path)

        # 如果没有加载到已有结果，初始化空列表
        if start_index == 0:
            self.results = []

        print(f"开始处理数据，从索引 {start_index} 开始...")

        # 创建事件循环
        loop = asyncio.get_event_loop()

        # 按批次处理数据，从start_index开始
        save_interval = 0
        num_genreate = self.config[self.step_name].get("num_genreate", 1)
        batch_size = self.config[self.step_name].get("batch_size", num_genreate)
        batch_size = batch_size // num_genreate
        if batch_size < 1:
            batch_size = 1
        print("开始生成...")
        print(f"num_genreate: {num_genreate}, batch_size: {batch_size}")
        for i in tqdm.tqdm(range(start_index, data_len, batch_size)):
            batch_items = data[i: i + batch_size]

            # 处理当前批次
            batch_results = loop.run_until_complete(
                self.process_batch(batch_items, image_dir)
            )

            # 为数据打上标签
            for item in batch_results:
                # 为原始数据打标签
                tags = self.tag_origin_answer_data(item)
                item.update(tags)

                # 为模型输出打标签
                tags = self.tag_output_data(item)
                item.update(tags)
                self.results.append(item)

            save_interval += 1
            # 每5个批次保存一次
            if save_interval >= self.config[self.step_name]["save_interval"]:
                save_interval = 0
                self.save_results(output_path)
                self.save_progress(i + batch_size, output_path)
                print(f"已处理 {i + batch_size} 条数据，已保存到 {output_path}")

            # 收到停止信号
            if stop_requested:
                self.save_results(output_path)
                self.save_progress(i + batch_size, output_path)
                print(
                    f"收到停止信号，已处理 {i + batch_size} 条数据，已保存到 {output_path}"
                )
                return

        # 保存最终结果
        self.save_results(output_path)
        # 处理完成后删除进度文件
        progress_file = output_path.replace(".jsonl", "_progress.json")
        if os.path.exists(progress_file):
            os.remove(progress_file)
        print(f"处理完成！共生成 {len(self.results)} 条数据，已保存到 {output_path}")

        return


class SmolvlmProcessor(DataProcessor):
    """使用smolvlm生成回答"""

    def __init__(self, config: dict):
        super().__init__(config, "smolvlm")

    # 预处理模型输入数据
    def preprocess_item(
            self, item: Dict[str, Any], image_dir: str
    ) -> Optional[Dict[str, Any]]:
        """预处理模型输入数据"""
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
            return None

        # 获取原始回答
        original_answer = item["conversations"][1]["content"]

        return {
            "item": item,
            "img_path": img_path,
            "prompt": prompt,
            "original_answer": original_answer,
        }

    async def request_model(self, processed_item: Dict[str, Any]) -> Optional[str]:
        """请求大模型生成DPO数据"""
        api_url = self.config[self.step_name]["api_url"] + "v1/chat/completions"
        model_name = self.config[self.step_name]["model_name"]
        temperature = self.config[self.step_name].get("temperature", 0.1)
        max_tokens = self.config[self.step_name].get("max_tokens", 8000)

        return await async_chat_with_llm(
            processed_item["img_path"],
            processed_item["prompt"],
            api_url,
            model_name,
            temperature,
            max_tokens,
        )

    def postprocess_result(
            self, processed_item: Dict[str, Any], model_response: List[str]
    ) -> Optional[Dict[str, Any]]:
        """后处理DPO结果"""
        return {
            "image": processed_item["item"]["image"],
            "prompt": processed_item["prompt"],
            "original_answer": processed_item["original_answer"],
            "data_type": self.config[self.step_name]["data_type"],
            "smolvlm_answers": model_response,
        }

    def tag_origin_answer_data(self, item: Dict[str, Any]) -> bool:
        """过滤输入数据"""
        return {
            "smorigin_answer_tags": {
                "repeat": check_repeat(item["original_answer"]),
                "short_words_repeat": has_many_short_words_sequence(
                    item["original_answer"]
                ),
            }
        }

    def tag_output_data(self, result_item: Dict[str, Any]) -> dict:
        """为输出数据打标签"""
        model_response = result_item["smolvlm_answers"]
        tags = {
            "smolvlm_tags": {
                "repeat": [check_repeat(r) for r in model_response],
                "short_words_repeat": [
                    has_many_short_words_sequence(r) for r in model_response
                ],
            }
        }
        return tags


# 使用SFT训练过的模型生成多条描述
def step1(jsonl_file_path, image_dir, config):
    """使用smolvlm生成回答"""
    processor = SmolvlmProcessor(config)
    return processor.process_data(jsonl_file_path, image_dir)


# 使用SFT训练过的模型生成多条问答
def step2(jsonl_file_path, image_dir, config):
    """使用smolvlm生成回答"""
    processor = SmolvlmProcessor(config)
    processor.step_name = "smolvlm_QA"
    return processor.process_data(jsonl_file_path, image_dir)


def format_dpo_data(jsonl_file_paths: list[str], image_dir, config):
    """格式化生成的DPO数据"""
    datas = []
    for jsonl_file in jsonl_file_paths:
        datas += load_jsonl(jsonl_file)
    output_path = config["format"]["output_path"]
    num_of_skip = 0

    dpo_format_datas = []
    for data in datas:
        if data["original_answer"] != data["smolvlm_answers"][0]:
            skip = False
            if data["data_type"] == "QA":
                or_answer = data["original_answer"].split(" ")
                # 原来的回答只有一个词,而且这个词在生成的答案中,那么跳过
                if len(or_answer) == 1 and or_answer[0] in data["smolvlm_answers"][0]:
                    skip = True
                    num_of_skip += 1
            if not skip:
                dpo_data = {
                    "prompt": data["prompt"],
                    "chosen": data["original_answer"],
                    "rejected": data["smolvlm_answers"][0],
                    "image": data["image"],
                    "data_type": data["data_type"],
                }
                dpo_format_datas.append(dpo_data)

        if data["original_answer"] != data["smolvlm_answers"][1]:
            skip = False
            if data["data_type"] == "QA":
                or_answer = data["original_answer"].split(" ")
                # 原来的回答只有一个词,而且这个词在生成的答案中,那么跳过
                if len(or_answer) == 1 and or_answer[0] in data["smolvlm_answers"][1]:
                    skip = True
                    num_of_skip += 1
            if not skip:
                dpo_data = {
                    "prompt": data["prompt"],
                    "chosen": data["original_answer"],
                    "rejected": data["smolvlm_answers"][1],
                    "image": data["image"],
                    "data_type": data["data_type"],
                }
                dpo_format_datas.append(dpo_data)

    save_jsonl(dpo_format_datas, output_path)
    print(f"共跳过 {num_of_skip} 条数据")
    print(f"格式化完成！共生成 {len(dpo_format_datas)} 条数据，已保存到 {output_path}")


if __name__ == "__main__":
    config = load_config(
        os.path.join(os.path.dirname(__file__), "generate_DPO_data_config.yaml")
    )
    image_dir = config["image_dir"]

    # step1(config["desc_data_input_jsonl"], image_dir, config)

    # step2(config["question_data_input_jsonl"], image_dir, config)

    # ---------------------------------------------------------------
    jsonl_file_paths = [config["smolvlm"]["output_path"],
                        config["smolvlm_QA"]["output_path"]]
    format_dpo_data(jsonl_file_paths, image_dir, config)
    # ---------------------------------------------------------------
