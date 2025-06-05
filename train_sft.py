"""
@File    :   train_mutil.py
@Time    :   2025/06/01 0:12:02
@Author  :   xco2
@Version :   1.0
当前正在运行的训练代码
"""
import unsloth
from unsloth import FastLanguageModel
import os
import torch
from torch import nn
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import yaml
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel

# ========== 加载配置文件 ==========
def load_config(config_path="config.yaml"):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

# ========== 配置部分 ==========
USE_LORA = config["use_lora"]  # 是否使用LoRA微调
SMOL = config["smol"]  # 是否使用小模型
MODEL_ID = config["model_id"] if SMOL else config["model_id_large"]

CHECKPOINT_FILE = config["checkpoint_file"]

# ========== 训练参数 ==========
model_name = MODEL_ID.split("/")[-1]
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["WANDB_API_KEY"] = config["wandb"]["api_key"]
os.environ["WANDB_PROJECT"] = config["wandb"]["project"]
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
    batch = lora_config["batch"]
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
    batch = lora_config["batch"]
    lr = lora_config["lr"]
    gradient_steps = lora_config["gradient_steps"]

# --------全参数微调---------(USE_LORA=False时生效)
LAST_N_TRAIN_LAYER = config["full_finetune"]["last_n_train_layer"]  # 要进行全量微调的最后几层

# 训练数据集路径
dataset_path = config["dataset"]["train_path"]
test_dataset_path = config["dataset"]["test_path"]

# 图片路径
image_dir = config["dataset"]["image_dir"]
bf_16 = config["training"]["bf16"]  # 是否使用bf16训练

RANDOM_SEED = config["training"]["random_seed"]

max_length = config["training"]["max_length"]

training_args = TrainingArguments(
    num_train_epochs=epoch,
    per_device_train_batch_size=batch,
    gradient_accumulation_steps=gradient_steps,
    warmup_steps=config["training"]["warmup_steps"],
    learning_rate=lr,
    lr_scheduler_type=config["training"]["lr_scheduler_type"],
    lr_scheduler_kwargs={"min_lr": config["training"]["lr_min"]},
    dataloader_num_workers=config["training"]["dataloader_num_workers"],  # 数据加载使用n进程
    max_grad_norm=config["training"]["max_grad_norm"],
    weight_decay=config["training"]["weight_decay"],
    logging_steps=config["training"]["logging_steps"],
    save_strategy=config["training"]["save_strategy"],
    save_steps=config["training"]["save_steps"],
    seed=RANDOM_SEED,
    # 最多保存的模型数量，超过该数量后，最早保存的模型会被删除
    save_total_limit=config["training"]["save_total_limit"],
    optim=config["training"]["optim"],  # 8bit用paged_adamw_8bit
    bf16=bf_16,
    output_dir=config["training"]["output_dir_template"].format(model_name=model_name),
    hub_model_id=config["training"]["hub_model_id_template"].format(model_name=model_name),
    # 是否移除数据集中未使用的列，设置为 False 可保留所有列
    remove_unused_columns=config["training"]["remove_unused_columns"],
    report_to=config["training"]["report_to"],
    run_name=config["training"]["run_name"],
    # 是否对数据加载器中的数据进行内存锁定，设置为 False 不进行内存锁定
    dataloader_pin_memory=config["training"]["dataloader_pin_memory"],
    # --------------测试------------------
    # eval_strategy=config["training"]["eval_strategy"],  # 每个epoch评估一次
    # per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
)

# ========== 模型 ==========
def build_model_unsloth():
    if USE_LORA:
        # base model
        model, processor = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=max_length,
            load_in_4bit=False,
            dtype=torch.bfloat16 if bf_16 else None,
            cache_dir=r"/hy-tmp/hub"
            # No|ne for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        )
        if CHECKPOINT_FILE is not None:
            # Load LoRA checkpoint
            model = PeftModel.from_pretrained(model, CHECKPOINT_FILE)
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
        # Load model
        model, processor = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=max_length,
            dtype=torch.bfloat16 if bf_16 else None,
            # No|ne for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        )
        # 如果只想微调LLM部分，冻结视觉模型参数
        for param in model.model.vision_model.parameters():
            param.requires_grad = False
        # 微调最后2层的参数
        last_n_layers = model.model.text_model.layers[:-LAST_N_TRAIN_LAYER]
        for layer in last_n_layers:
            for param in layer.parameters():
                param.requires_grad = False
        # 微调connector的参数
        model.model.connector.modality_projection.proj.weight.requires_grad = True
    return processor, model


def load_processor_and_model():
    if __name__ == "__main__":
        return build_model_unsloth()
    else:
        return AutoProcessor.from_pretrained(MODEL_ID), None


def print_model_info(model):
    print(model)
    peak_mem = torch.cuda.max_memory_allocated()
    print(f"模型当前占用显存: {peak_mem / 1024 ** 3:.2f} GB")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable_params / 1e5:.2f} 万")
    print(f"总参数: {total_params / 1e5:.2f} 万")


def load_dataset_fn():
    print("加载数据集...")
    raw_dataset = load_dataset("json", data_files={"train": dataset_path, "test": test_dataset_path})
    train_ds = raw_dataset["train"]
    test_ds = raw_dataset["test"]
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


def collate_fn(examples):
    instances = []
    for example in examples:
        image = os.path.join(image_dir, example["image"].replace("\\", "/"))
        question = example["conversations"][0]["content"].replace("\n<image>", "")
        # 去除问题中的字数限制提示
        if question.startswith("Write"):
            question = question.replace("100 words ", "")
        elif question.startswith("Analyze"):
            question = question.replace(" Keep it 100 words.", "")
        answer = example["conversations"][1]["content"]

        user_content = [{"type": "text", "text": question}]
        user_content.append({"type": "image", "path": image})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]

        instance = processor.apply_chat_template(messages, add_generation_prompt=False,
                                                 tokenize=True, return_dict=True, return_tensors="pt")
        instances.append(instance)

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
        "labels": labels
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


from transformers.trainer import *
from transformers.trainer import _is_peft_model


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        # 去掉inputs中的num_items_in_batch
        if 'num_items_in_batch' in inputs:
            inputs.pop('num_items_in_batch')

        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
                self.args.average_tokens_across_devices
                and (self.model_accepts_loss_kwargs or self.compute_loss_func)
                and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


def train_main():
    # processor, model = load_processor_and_model()
    print_model_info(model)
    train_ds, test_ds = load_dataset_fn()
    # image_token_id = get_image_token_id(processor)
    # collate_fn = collate_fn_builder(processor, model.dtype, image_dir, image_token_id)

    # def compute_metrics(eval_pred):
    #     print("计算指标...")
    #     logits, labels = eval_pred
    #     print("logits:", logits.shape)
    #     print("labels:", labels.shape)
    #     predictions = np.argmax(logits, axis=1)
    #     # loss = model.compute_loss(labels=labels, inputs=eval_pred[0]).item()
    #     return {"accuracy": (predictions == labels).mean()}#, "loss": loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_ds,
        # eval_dataset=test_ds,
        # compute_metrics=compute_metrics,
    )
    print("开始训练...")
    trainer.train()
    print("训练完成！")
    # eval_results = trainer.evaluate()
    # print("评估结果：", eval_results)


if __name__ == "__main__":
    train_main()
