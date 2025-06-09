from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel, LoraConfig
import torch
import os
import json
import pandas as pd
from transformers import TextStreamer
from PIL import Image
import numpy as np
import inspect
import yaml


class AttentionViewer:
    def __init__(self, config_path="attention_view_config.yaml"):
        """
        初始化 AttentionViewer 类
        
        Args:
            config_path: YAML配置文件路径
        """
        # 加载配置文件
        self.config = self._load_config(config_path)
        
        # 设置环境变量
        os.environ["HF_ENDPOINT"] = self.config["environment"]["hf_endpoint"]

        # 设置模型路径
        self.base_model_path = self.config["model"]["base_model_path"]
        self.lora_step = self.config["model"]["lora_step"]
        self.SMOL = self.config["model"]["SMOL"]

        # 如果未提供lora_adapter_path，则自动生成
        lora_adapter_path = self.config["model"]["lora_adapter_path"]
        if lora_adapter_path is None and self.lora_step > 0:
            if self.SMOL:
                template = self.config["model"]["small_lora_path_template"]
                self.lora_adapter_path = template.format(lora_step=self.lora_step)
            else:
                template = self.config["model"]["large_lora_path_template"]
                self.lora_adapter_path = template.format(lora_step=self.lora_step)
                # 暂不支持2B模型的attention可视化
                raise NotImplementedError("2B模型的attention可视化暂不支持")
        else:
            self.lora_adapter_path = lora_adapter_path

        self.save_path = self.config["output"]["save_path"]
        os.makedirs(self.save_path, exist_ok=True)

        # 加载处理器和模型
        self.processor = AutoProcessor.from_pretrained(self.base_model_path)
        self.load_model()

        # 打印模型信息
        self.print_model_info()
    
    def _load_config(self, config_path):
        """加载YAML配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def load_model(self):
        """加载模型，根据lora_step决定是否使用LoRA"""
        # 加载基础模型
        base_model_for_lora = AutoModelForImageTextToText.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            # _attn_implementation="flash_attention_2"
        )

        if self.lora_step > 0:
            print("加载LORA...")
            # 加载LoRA适配器到基础模型
            model = PeftModel.from_pretrained(base_model_for_lora, self.lora_adapter_path)

            # 将最终的模型移动到CUDA设备
            model = model.to("cuda").to(torch.bfloat16)

            # 合并LoRA权重到基础模型
            model = model.merge_and_unload().eval()
        else:
            print("使用基础模型...")
            model = base_model_for_lora
            model = model.to("cuda").to(torch.bfloat16).eval()

        self.model = model

    def print_model_info(self):
        """打印模型信息，包括显存占用、参数量等"""
        peak_mem = torch.cuda.max_memory_allocated()
        print(f"模型当前占用显存: {peak_mem / 1024 ** 3:.2f} GB")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"总参数: {total_params / 1e5:.2f} 万")
        print("-" * 60)

        print("model")
        print(type(self.model))
        file_path = inspect.getfile(self.model.__class__)
        print(file_path)
        print("-" * 60)
        print("text model")
        print(type(self.model.model.text_model))
        file_path = inspect.getfile(self.model.model.text_model.__class__)
        print(file_path)
        print("-" * 60)

    def generate_with_attention(self, image_path, prompt_text=None):
        """
        生成带有注意力信息的文本
        
        Args:
            image_path: 图像路径
            prompt_text: 提示文本，如果为None则使用默认提示
            
        Returns:
            generated_texts: 生成的文本
            attentions: 注意力信息
        """
        if prompt_text is None:
            prompt_text = self.config["generation"]["default_prompt"]

        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image_path},
                        {"type": "text", "text": prompt_text},
                    ]
                },
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)

            print(f'[Image]: {image_path}')

            # 从配置中获取生成参数
            gen_config = self.config["generation"]
            generated_ids = self.model.generate(
                **inputs,
                num_beams=gen_config["num_beams"],
                max_length=gen_config["max_length"],
                early_stopping=gen_config["early_stopping"],
                no_repeat_ngram_size=gen_config["no_repeat_ngram_size"],
                length_penalty=gen_config["length_penalty"],
                output_attentions=True,
                return_dict_in_generate=True
            )

            attentions = generated_ids.attentions
            self._print_attention_info(attentions)

            # 解码生成的文本
            generated_texts = self.processor.batch_decode(
                generated_ids.sequences,
                skip_special_tokens=True,
            )
            print(f'🤖️: {generated_texts}')
            print("-" * 50)

            return generated_texts, generated_ids

    def _print_attention_info(self, attentions):
        """打印注意力信息"""
        att_len_str = f"{len(attentions)} 字"
        print(att_len_str)
        if len(attentions) > 0:
            layer_str = f"{len(attentions[0])} 层"
            print(layer_str)
            if len(attentions[0]) > 0:
                att_map_str = f"att map: {attentions[0][0].shape}"
                print(att_map_str)

    def process_attention_maps(self, generated_ids):
        """
        处理注意力图
        
        Args:
            generated_ids: 生成的token IDs和注意力信息
            
        Returns:
            block_index: 图像块索引信息
            max_row: 最大行数
            max_col: 最大列数
            attentions: 注意力信息
        """
        attentions = generated_ids.attentions
        all_tokens = generated_ids.sequences[0].cpu().numpy()

        # 找切割为横竖多少个格子
        # resize成512
        # 一张图变8*8个token
        # 所以一个token对应512/8=64
        # 所以一个格子对应64*64个像素
        special_tokens = [f"<row_{i}_col_{j}>" for i in range(1, 5) for j in range(1, 5)]
        special_tokens += ["<fake_token_around_image>", "<global-img>"]

        image_special_tokens = {k: self.processor.tokenizer.added_tokens_encoder[k] for k in special_tokens}
        fake_tokens_index = np.where(all_tokens == image_special_tokens["<fake_token_around_image>"])[0]

        block_index = []
        # 计算每个block的位置
        max_row = 0
        max_col = 0
        prompt_start_index = fake_tokens_index[-1] + 1

        # 处理每个图像块
        for f in fake_tokens_index[:-1]:
            position_token = all_tokens[f + 1]
            row = None
            col = None
            for k in image_special_tokens.keys():
                if image_special_tokens[k] == position_token:
                    if k.startswith("<row_"):
                        k = k.replace(">", "")
                        rc = k.split("_")
                        row = int(rc[1])
                        col = int(rc[3])
                        max_row = max(max_row, row)
                        max_col = max(max_col, col)
                    else:
                        row = -1
                        col = -1
                    break
            block_index.append({"row": row, "col": col, "start": f + 2, "end": f + 66})

        # 收集每个block的attention map
        for token_index, layer_attention in enumerate(attentions):
            for layer, attention_map in enumerate(layer_attention):
                attention_map = attention_map.to(dtype=torch.float32).cpu().numpy()
                for bi in block_index:
                    start = bi["start"]
                    end = bi["end"]
                    att = attention_map[0, :, prompt_start_index:, start:end]
                    att = np.sum(att, axis=(0, 1))
                    if "att" not in bi:
                        bi["att"] = {}
                    if layer not in bi["att"]:
                        bi["att"][layer] = {}
                    bi["att"][layer].update({token_index: att})

        return block_index, max_row, max_col, attentions

    def create_attention_visualizations(self, block_index, max_row, max_col, attentions, image_path):
        """
        创建注意力可视化图像
        
        Args:
            block_index: 图像块索引信息
            max_row: 最大行数
            max_col: 最大列数
            attentions: 注意力信息
            image_path: 原始图像路径
        """
        # 从配置中获取alpha值
        alpha = self.config["output"]["alpha"]
        
        # 合并同一层的attention map
        for bi in block_index:
            att = bi["att"]
            for layer, att_map in att.items():
                layer_attention_map = np.zeros(64)
                for token_index, attention_map in att_map.items():
                    layer_attention_map += attention_map
                layer_attention_map = layer_attention_map.reshape(8, 8)
                # layer_attention_map = np.log(layer_attention_map + 1)
                layer_attention_map = layer_attention_map / np.max(layer_attention_map)
                layer_attention_map = layer_attention_map * 255
                layer_attention_map = layer_attention_map.astype(np.uint8)
                layer_attention_map = Image.fromarray(layer_attention_map)
                layer_attention_map = layer_attention_map.resize((512, 512), Image.NEAREST)
                layer_attention_map = np.array(layer_attention_map)
                if "view" not in bi:
                    bi["view"] = {}
                bi["view"][layer] = layer_attention_map

        # 拼接图片
        max_layer = len(attentions[0])
        block_size = 512
        block_layer_view = [np.zeros((2048, 2048)) for _ in range(max_layer)]
        whole_layer_view = []

        for bi in block_index:
            row = bi["row"] - 1
            col = bi["col"] - 1
            att = bi["view"]
            for layer, layer_attention_map in att.items():
                if row < 0 and col < 0:
                    whole_layer_view.append(layer_attention_map)
                    continue
                block_layer_view[layer][row * block_size:(row + 1) * block_size,
                col * block_size:(col + 1) * block_size] = layer_attention_map

        # 把map盖到原图上,并保存图片
        image = Image.open(image_path)
        image = image.resize((max_col * block_size, max_row * block_size))
        image = np.array(image)

        # 保存块层视图
        self._save_block_layer_views(block_layer_view, image, max_row, max_col, block_size, alpha)

        # 保存整体层视图
        self._save_whole_layer_views(whole_layer_view, image, alpha)

    def _save_block_layer_views(self, block_layer_view, image, max_row, max_col, block_size, alpha):
        """保存块层视图"""
        for layer, layer_attention_map in enumerate(block_layer_view):
            layer_attention_map = layer_attention_map.astype(np.float32)
            save_image = np.zeros((layer_attention_map.shape[0], layer_attention_map.shape[1], 3))
            save_image[:image.shape[0], :image.shape[1], :] = image
            save_image[:, :, 0] = (1 - alpha) * save_image[:, :, 0] + alpha * layer_attention_map
            save_image[:, :, 1:] = (1 - alpha) * save_image[:, :, 1:]

            # 添加蓝色边界线（每512像素一条）
            for i in range(1, max_row):
                save_image[i * block_size - 1:i * block_size + 1, :, 0] = 0
                save_image[i * block_size - 1:i * block_size + 1, :, 1] = 0
                save_image[i * block_size - 1:i * block_size + 1, :, 2] = 255  # 蓝色
            for j in range(1, max_col):
                save_image[:, j * block_size - 1:j * block_size + 1, 0] = 0
                save_image[:, j * block_size - 1:j * block_size + 1, 1] = 0
                save_image[:, j * block_size - 1:j * block_size + 1, 2] = 255  # 蓝色

            # 添加绿色边界线（每64像素一条）
            pixel_per_token = 64
            for row in range(max_row):
                for col in range(max_col):
                    for i in range(1, 8):  # 每个大块内有8个小块
                        # 水平绿线
                        y = row * block_size + i * pixel_per_token
                        if y < save_image.shape[0]:
                            save_image[y, col * block_size:(col + 1) * block_size, 0] = 0
                            save_image[y, col * block_size:(col + 1) * block_size, 1] = 255  # 绿色
                            save_image[y, col * block_size:(col + 1) * block_size, 2] = 0

                        # 垂直绿线
                        x = col * block_size + i * pixel_per_token
                        if x < save_image.shape[1]:
                            save_image[row * block_size:(row + 1) * block_size, x, 0] = 0
                            save_image[row * block_size:(row + 1) * block_size, x, 1] = 255  # 绿色
                            save_image[row * block_size:(row + 1) * block_size, x, 2] = 0

            save_image = save_image.astype(np.uint8)
            save_image = Image.fromarray(save_image)
            save_image.save(os.path.join(self.save_path, f"block_layer_view_{layer}.png"))

    def _save_whole_layer_views(self, whole_layer_view, image, alpha):
        """保存整体层视图"""
        for layer, layer_attention_map in enumerate(whole_layer_view):
            layer_attention_map = layer_attention_map.astype(np.uint8)
            layer_attention_map = Image.fromarray(layer_attention_map)
            layer_attention_map = layer_attention_map.resize((image.shape[1], image.shape[0]), Image.NEAREST)
            layer_attention_map = np.array(layer_attention_map)
            save_image = image.copy()
            save_image[:, :, 0] = (1 - alpha) * save_image[:, :, 0] + alpha * layer_attention_map
            save_image[:, :, 1:] = (1 - alpha) * save_image[:, :, 1:]
            save_image = save_image.astype(np.uint8)
            save_image = Image.fromarray(save_image)
            save_image.save(os.path.join(self.save_path, f"whole_layer_view_{layer}.png"))

    def process_image(self, image_path, prompt_text=None):
        """
        处理图像并生成注意力可视化
        
        Args:
            image_path: 图像路径
            prompt_text: 提示文本，如果为None则使用默认提示
        """
        # 获取图片名称（不含后缀）并创建对应文件夹
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_save_path = os.path.join(self.save_path, str(self.lora_step), image_name)
        os.makedirs(image_save_path, exist_ok=True)

        # 临时保存原始save_path
        original_save_path = self.save_path
        # 修改save_path为新创建的文件夹路径
        self.save_path = image_save_path

        # 生成带有注意力的文本
        _, generated_ids = self.generate_with_attention(image_path, prompt_text)

        # 处理注意力图
        block_index, max_row, max_col, attentions = self.process_attention_maps(generated_ids)

        # 创建注意力可视化
        self.create_attention_visualizations(block_index, max_row, max_col, attentions, image_path)

        # 恢复原始save_path
        self.save_path = original_save_path

    def process_config_images(self):
        # 从配置文件中获取测试图像路径
        image_paths = self.config["test_images"]

        # 处理每张图像
        for image_path in image_paths:
            self.process_image(image_path)

    def __del__(self):
        """析构函数，释放显存"""
        if hasattr(self, 'model'):
            del self.model


# 示例用法
if __name__ == "__main__":
    # 创建AttentionViewer实例，使用配置文件
    viewer = AttentionViewer("attention_view_config.yaml")

    viewer.process_config_images()
