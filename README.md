# smolvlm2-500M-illustration-description 🎨

<p align="center">
    🤗<a href="https://huggingface.co/Qwen">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤗<a href="https://huggingface.co/spaces/xco2/smolvlm2-500M-illustration-description">Demo</a>&nbsp&nbsp | &nbsp&nbsp<a href="README_zh.md">简体中文</a>
</p>

An illustration description generation model that provides more detailed and vivid descriptions of the images ✨

This model is fine-tuned based on HuggingFaceTB/SmolVLM2-500M-Video-Instruct.

## Usage 🚀
This model can be used to generate descriptions of illustrations and answer some simple questions related to the content of the illustrations.

### Prompt Suggestions 💡
- Write a descriptive caption for this image in a formal tone.
- Write a descriptive caption for this image in a casual tone.
- Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.
- What color is the hair of the character?
- What are the characters wearing?

## Training Details 📋

### Training Code 🛠️
Use `train_sft.py` for training, and configure the relevant parameters in `config.yaml`.

### Training Dataset 📊
1. Use the quantized fancyfeast/joy-caption-pre-alpha model to generate descriptions for approximately 100,000 illustrations using multiple prompts.
2. Filter out the meaningless descriptions with repeated short phrases generated by the model.
3. Use Qwen3-12B to generate question-answer data related to the illustration content based on the generated illustration descriptions. The generation code can be found in `more_data_generate/generate_question.py`.

#### Finally, approximately 240,000 training data samples are obtained.


### Code File Description 📄
- `train_sft.py`: Supervised Fine-Tuning (SFT) training script
- `config.yaml`: Hyperparameter configuration file for model training
- `eval/eval.py`: Generates image descriptions and evaluates them using a reference LLM
- `eval/attention_view.py`: Visualizes model attention maps over input images
- `more_data_generate/generate_question.py`: Generates question-answer pairs from image descriptions
- `more_data_generate/generate_multi_results.py`: Generates diverse descriptions for DPO preference dataset construction
- 
## TODO 📝
- [ ] Train the model using reinforcement learning to improve the generation quality and reduce model hallucinations.

## Quick Start ⚡
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