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
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from trl import ORPOTrainer, ORPOConfig


# ========== 加载配置文件 ==========
def load_config(config_path="orpo_config.yaml"):
    """加载YAML配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


config = load_config()

# ========== 配置部分 ==========
USE_LORA = config["use_lora"]  # 是否使用LoRA微调
SMOL = config["smol"]  # 是否使用小模型
MODEL_ID = config["model_id"] if SMOL else config["model_id_large"]

# ========== 训练参数 ==========
model_name = MODEL_ID.split("/")[-1]
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"]
os.environ["WANDB_PROJECT"] = config["wandb"]["project"]
if config["wandb"]["mode"] == "online":
    os.environ["WANDB_API_KEY"] = config["wandb"]["api_key"]
else:
    os.environ["WANDB_MODE"] = config["wandb"]["mode"]

# --------LoRA 配置----------
if SMOL:
    lora_config = config["smol_lora"]
    target_modules_layer = [
        31 - i for i in range(lora_config["target_modules_layer"])
    ]  # 要进行 LoRA 微调的层索引
    target_modules_layer = target_modules_layer[::-1]
    LORA_R = lora_config["lora_r"]  # LoRA 中的 R 参数
    LM_HEAD_RANk = lora_config["lm_head_rank"]  # lm_head层的rank
    CONNECT_RANK = lora_config["connect_rank"]  # connector层的rank

    target_modules_lora_r = lora_config["target_modules_lora_r"]  # 每个层的 R 参数
    LORA_RANKS = {
        layer_idx: lora_r
        for layer_idx, lora_r in zip(target_modules_layer, target_modules_lora_r)
    }

    epoch = lora_config["epoch"]
    batch_size = lora_config["batch_size"]
    lr = lora_config["lr"]
    gradient_steps = lora_config["gradient_steps"]
else:
    lora_config = config["large_lora"]
    target_modules_layer = [
        23 - i for i in range(lora_config["target_modules_layer"])
    ]  # 要进行 LoRA 微调的层索引
    target_modules_layer = target_modules_layer[::-1]
    LORA_R = lora_config["lora_r"]  # LoRA 中的 R 参数
    LM_HEAD_RANk = lora_config["lm_head_rank"]  # lm_head层的rank
    CONNECT_RANK = lora_config["connect_rank"]  # connector层的rank

    # 对于大模型，使用动态计算每层的R参数
    target_modules_lora_r = lora_config["target_modules_lora_r"]  # 每个层的 R 参数
    LORA_RANKS = {
        layer_idx: lora_r
        for layer_idx, lora_r in zip(target_modules_layer, target_modules_lora_r)
    }

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

max_length = config["training"]["max_length"]

training_args = ORPOConfig(
    num_train_epochs=epoch,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_steps,
    warmup_steps=config["training"]["warmup_steps"],
    learning_rate=lr,
    lr_scheduler_type=config["training"]["lr_scheduler_type"],
    lr_scheduler_kwargs={"min_lr": config["training"]["lr_min"]},
    dataloader_num_workers=config["training"][
        "dataloader_num_workers"
    ],  # 数据加载使用n进程
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
    hub_model_id=config["training"]["hub_model_id_template"].format(
        model_name=model_name
    ),
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
def build_model():
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
        if CONNECT_RANK > 0:
            target_modules.append("model.connector.modality_projection.proj")
        if LM_HEAD_RANk > 0:
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
        if CONNECT_RANK > 0:
            rank_pattern["model.connector.modality_projection.proj"] = CONNECT_RANK
        if LM_HEAD_RANk > 0:
            rank_pattern["lm_head"] = LM_HEAD_RANk

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
        return build_model()
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
    data_files = {"train": dataset_path}
    if test_dataset_path is not None:
        data_files["test"] = test_dataset_path

    raw_dataset = load_dataset(
        "json", data_files=data_files
    )
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
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]


processor, model = load_processor_and_model()
image_token_id = get_image_token_id(processor)
dtype = torch.bfloat16 if bf_16 or model is None else model.dtype


def collate_fn(examples):
    choicen_instances = []
    rejected_instances = []

    for example in examples:
        image = os.path.join(image_dir, example["image"].replace("\\", "/"))
        question = example["prompt"]
        choicen = example["chosen"]
        rejected = example["rejected"]

        user_content = [{"type": "text", "text": question},
                        {"type": "image", "path": image}]

        choicen_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": choicen}]},
        ]

        rejected_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": rejected}]}
        ]

        choicen_instance = processor.apply_chat_template(
            choicen_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        rejected_instance = processor.apply_chat_template(
            rejected_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        choicen_instances.append(choicen_instance)
        rejected_instances.append(rejected_instance)

    len_chosen = len(choicen_instances)
    instances = choicen_instances + rejected_instances

    input_ids = pad_sequence(
        [inst["input_ids"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id,
    )
    attention_mask = pad_sequence(
        [inst["attention_mask"].squeeze(0) for inst in instances],
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        [inst["input_ids"].squeeze(0).clone() for inst in instances],
        batch_first=True,
        padding_value=-100,
    )

    labels[labels == image_token_id] = -100

    out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "len_chosen": len_chosen}

    # Step 1: figure out maximum frames, height, width across the batch
    pvs = [
        inst["pixel_values"].squeeze(0) for inst in instances if "pixel_values" in inst
    ]
    if pvs:  # there is at least one non-None pixel_values
        max_frames = max(pv.shape[0] for pv in pvs)
        max_h = max(pv.shape[-2] for pv in pvs)
        max_w = max(pv.shape[-1] for pv in pvs)
    else:
        max_h = max_w = processor.video_size["longest_edge"]
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
                (max_frames, c, max_h, max_w), dtype=pv.dtype, device=pv.device
            )
            padded_pv[:f, :, :h, :w] = pv
        padded_pixel_values_list.append(padded_pv)

    out["pixel_values"] = torch.stack(padded_pixel_values_list, dim=0)
    return out


from trl.trainer.orpo_trainer import *


class CustomTrainer(ORPOTrainer):

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> dict:
        return feature

    def concatenated_forward(
            self, model: nn.Module, batch: dict[str, Union[int, torch.LongTensor]]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        # 重写这个部分对数据的预处理
        len_chosen = batch.pop("len_chosen")

        model_kwargs = batch

        # model_kwargs["output_router_logits"] = True

        outputs = model(
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = batch["labels"].clone()
        attention_mask = batch["attention_mask"]
        # labels = torch.where(attention_mask == 1, labels, self.label_pad_token_id)
        # orpo chosen nll loss is computed over the full prompt and response
        chosen_nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        all_logps = self.get_batch_logps(
            all_logits,
            batch["labels"],
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        if not self.is_encoder_decoder:
            chosen_logits = all_logits[:len_chosen, :-1, :]
            rejected_logits = all_logits[len_chosen:, :-1, :]
        else:
            chosen_logits = all_logits[:len_chosen]
            rejected_logits = all_logits[len_chosen:]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_nll_loss, outputs.aux_loss)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_nll_loss)


def train_main():
    print_model_info(model)
    train_ds, test_ds = load_dataset_fn()

    processor.pad_token_id = processor.tokenizer.eos_token_id

    trainer = CustomTrainer(
        model=model,
        processing_class=processor,
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
