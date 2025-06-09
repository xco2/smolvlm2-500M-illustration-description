# smolvlm2-500M-illustration-description 🎨

<p align="center">
          🤗<a href="https://huggingface.co/Qwen">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤗<a href="https://huggingface.co/spaces/xco2/smolvlm2-500M-illustration-description">Demo</a>&nbsp&nbsp
</p>

一个插画描述生成模型，提供更丰富的画面描述✨

模型基于HuggingFaceTB/SmolVLM2-500M-Video-Instruct进行微调

## 用法 🚀
该模型可用于生成插画的描述，与进行一些简单的插画内容有关的问答

### prompt建议 💡
- Write a descriptive caption for this image in a formal tone.
- Write a descriptive caption for this image in a casual tone.
- Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.
- What color is the hair of the character?
- What are the characters wearing?

## 训练细节 📋

### 训练代码 🛠️
使用`train_sft.py`进行训练，相关参数在`config.yaml`中配置

### 训练数据集 📊
1. 使用量化后的fancyfeast/joy-caption-pre-alpha模型，使用多个prompt对约100000张插画进行描述
2. 过滤掉模型重复短句的无意义描述
3. 使用qwen3-12B根据生成的插画描述，生成与插画内容相关的问答数据，生成代码见`more_data_generate/generate_question.py`

#### 最后得到约24万条训练数据

### 代码文件说明 📄
- `train_sft.py`：训练脚本
- `config.yaml`：训练参数配置
- `eval/eval.py`：使用指定文件生成描述，并使用更大的模型进行评分
- `eval/attention_view.py`：可视化模型在输入图片上的注意力分布
- `more_data_generate/generate_question.py`：根据生成的描述生成问答数据
- `more_data_generate/generate_multi_results.py`：用于对同一张插画生成多个描述数据的脚本，后续可用于构造DPO训练数据

## TODO 📝
- [ ] 使用强化学习训练模型，尝试提升生成效果，减少模型幻觉

## 快速使用 ⚡
```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
import torch

model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
adapter_name = "xco2/smolvlm2-500M-illustration-description"

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2"
)
model = PeftModel.from_pretrained(model, adapter_name)

processor = AutoProcessor.from_pretrained(model_name)

model = model.to('cuda').to(torch.bfloat16)
model = model.merge_and_unload().eval()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image",
             "url": "https://cdn.donmai.us/sample/63/e7/__castorice_honkai_and_1_more_drawn_by_yolanda__sample-63e73017612352d472b24056e501656d.jpg"},
            {"type": "text",
             "text": "Write a descriptive caption for this image in a formal tone."},
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=model.dtype)

generated_ids = model.generate(**inputs, do_sample=True, max_new_tokens=2048)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
print("Assistant:", generated_texts[0].split("Assistant:")[-1])
```