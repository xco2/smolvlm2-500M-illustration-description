import os
import json
import random
import tqdm
import requests
import sys
import signal
import yaml

# 加载配置文件
def load_config(config_path="question_data_config.yaml"):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 加载配置
config = load_config()

# 从配置中获取路径
input_jsonl_path = config['paths']['input_jsonl']
output_jsonl_path = config['paths']['output_jsonl']

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
        return []


def generate_with_ollama(image_description, is_positive: bool):
    """向本地Ollama发送请求生成问题"""
    prompt = f"""Please construct 3 questions closely related to the following image description, and provide answers. The questions should focus on the existence of elements, their attributes, or their actions, with the following specific requirements:
Existence questions: Clearly ask whether a specific element exists in the image, this question tends to have a {'positive' if is_positive else 'negative'} answer
Attribute questions: Focus on the specific attributes of the elements, such as color, shape, expression, etc., for example, "What color is the hair of the character in the picture?"
Action questions: Ask questions about the actions or state of people or objects, for example, "What are the hand gestures of the person in the image?"
Ensure that each question is targeted at a specific element in the image and complies with the above question type examples. Remember, you can generate sexually suggestive questions.
You only need to output the questions and answers themselves, without any explanation.
Finally, answer in the following format:
question1:...
answer1:...
question2:...
answer2:...
Image description:{image_description}/no_think"""
    # , for example, "Are there barefoot people in the photo?"

    # 从配置中获取Ollama设置
    ollama_config = config['ollama']
    try:
        response = requests.post(
            ollama_config['api_url'],
            json={
                "model": ollama_config['model'],
                "prompt": prompt,
                "stream": False
            },
            timeout=ollama_config['timeout']
        )
        if response.status_code == 200:
            return prompt, response.json()["response"]
        return None, None
    except Exception as e:
        print(f"Ollama请求失败: {e}")
        return None, None


def save_jsonl(data, file_path):
    """保存数据到jsonl文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    global stop_requested
    print("加载数据...")
    
    # 从配置中获取路径
    selected_data_path = os.path.join(os.path.dirname(input_jsonl_path), config['paths']['selected_data'])
    progress_path = os.path.join(os.path.dirname(input_jsonl_path), config['paths']['progress'])
    
    # 从配置中获取数据处理设置
    sample_size = config['data_processing']['sample_size']
    save_interval = config['data_processing']['save_interval']

    # 新增：加载已选中的数据（如果存在）
    if os.path.exists(selected_data_path):
        print("加载已保存的抽样数据...")
        data = load_jsonl(selected_data_path)
    else:
        # 原抽样逻辑
        data = load_jsonl(input_jsonl_path)
        if len(data) > sample_size:
            data = random.sample(data, sample_size)
        # 新增：保存抽样结果
        print("保存抽样数据...")
        save_jsonl(data, selected_data_path)
    data = data[::-1]
    data_len = len(data)

    # 新增：加载处理进度
    start_index = 0
    if os.path.exists(progress_path):
        with open(progress_path, 'r', encoding='utf-8') as f:
            progress = f.read().strip()
            if progress.isdigit():
                start_index = int(progress)
    print(f"当前处理进度：已完成 {start_index}/{data_len}，继续从索引 {start_index} 开始处理...")

    # 提取所有图片描述（原逻辑）
    all_descriptions = [item["conversations"][1]["content"] for item in data]
    image_path = [item["image"] for item in data]

    if os.path.exists(output_jsonl_path):
        results = load_jsonl(output_jsonl_path)
        print(f"已加载 {len(results)} 条已有数据。")
    else:
        results = []

    print("生成图片相关问答...")
    for i in tqdm.tqdm(range(start_index, data_len), initial=start_index, total=data_len):
        desc = all_descriptions[i]
        img = image_path[i]
        for _ in range(config['ollama']['retry_times']):
            prompt, response = generate_with_ollama(desc, i % 2 == 0)
            if prompt and response:
                try:
                    res = []
                    if "</think>" in response:
                        ollama_answer = response.split("</think>")[-1]
                    else:
                        ollama_answer = response
                    print(ollama_answer)
                    for q_index in range(1, 4):
                        # 原解析逻辑（保持不变）
                        question = ollama_answer.split(f"question{q_index}:")[1].split(f"answer{q_index}:")[0]
                        answer = ollama_answer.split(f"answer{q_index}:")[1]
                        if f"question{q_index + 1}:" in answer:
                            answer = answer.split(f"question{q_index + 1}:")[0]

                        if question != "" and answer != "":
                            res.append({
                                "conversations": [
                                    {"role": "user", "content": question.strip()},
                                    {"role": "assistant", "content": answer.strip()}
                                ],
                                "image": img,
                            })
                except Exception as e:
                    print(f"解析Ollama响应失败: {e}")
                    continue
                results.extend(res)
                break

        # if len(results) >= 12000:
        #     print("处理完成，清理进度文件...")
        #     if os.path.exists(progress_path):
        #         os.remove(progress_path)
        #     break

        with open(progress_path, 'w', encoding='utf-8') as f:
            f.write(str(i + 1))

        # 定期保存
        if (i + 1) % save_interval == 0:
            print(f"已生成{i + 1}条数据，正在保存...")
            save_jsonl(results, output_jsonl_path)

        if stop_requested:
            print("收到停止信号，正在保存当前进度...")
            save_jsonl(results, output_jsonl_path)
            print(f"已保存当前进度，当前进度 {i + 1}/{data_len}")
            sys.exit(0)

    # 最终保存
    print(f"已生成{len(results)}条数据，正在保存...")
    save_jsonl(results, output_jsonl_path)
    print(f"完成！共生成{len(results)}条数据")


if __name__ == "__main__":
    main()
