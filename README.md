# smolvlm2-500M-illustration-description 🎨

<p align="center">
    🤗<a href="https://huggingface.co/Qwen">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤗<a href="https://huggingface.co/spaces/xco2/smolvlm2-500M-illustration-description">Demo</a>&nbsp&nbsp | &nbsp&nbsp<a href="README_zh.md">简体中文</a>
</p>

An illustration description generation model that provides more detailed and rich descriptions of the pictures ✨

This model is fine-tuned based on HuggingFaceTB/SmolVLM2-500M-Video-Instruct.

## Usage 🚀
This model can be used to generate descriptions of illustrations and answer some simple questions related to the content of the illustrations.

### Prompt suggestions 💡
- Write a descriptive caption for this image in a formal tone.
- Write a descriptive caption for this image in a casual tone.
- Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.
- What color is the hair of the character?
- What are the characters wearing?

## Training details 📋

### Training dataset 📊
1. Use the quantized fancyfeast/joy-caption-pre-alpha model to generate descriptions for approximately 100,000 illustrations using multiple prompts 🖼
2. Filter out meaningless descriptions with repeated short phrases from the model.
3. Use Qwen3-12B to generate Q&A data related to the illustration content based on the generated illustration descriptions.
Finally, approximately 240,000 training data points are obtained.

## To-do list 📝
- [ ] Filter the training dataset using more methods and train the model with higher-quality data.
- [ ] Train the model using reinforcement learning to improve the generation effect and reduce model hallucinations.