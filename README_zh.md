# smolvlm2-500M-illustration-description 🎨

<p align="center">
          🤗<a href="https://huggingface.co/Qwen">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤗<a href="https://huggingface.co/spaces/xco2/smolvlm2-500M-illustration-description">Demo</a>&nbsp&nbsp
</p>

一个插画描述生成模型，提供更丰富的画面描述✨

基于HuggingFaceTB/SmolVLM2-500M-Video-Instruct进行微调

## 用法 🚀
该模型可用于生成插画的描述，与进行一些简单的插画内容有关的问答

### prompt建议 💡
- Write a descriptive caption for this image in a formal tone.
- Write a descriptive caption for this image in a casual tone.
- Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.
- What color is the hair of the character?
- What are the characters wearing?

## 训练细节 📋

### 训练数据集 📊
1. 使用量化后的fancyfeast/joy-caption-pre-alpha模型，使用多个prompt对约100000张插画进行描述🖼
2. 过滤掉模型重复短句的无意义描述
3. 使用qwen3-12B根据生成的插画描述，生成与插画内容相关的问答数据
最后得到约24万条训练数据

## 待办事项 📝
- [ ] 使用更多方式过滤训练数据集，使用更高质量的数据训练
- [ ] 使用强化学习训练模型，提升生成效果，减少模型幻觉