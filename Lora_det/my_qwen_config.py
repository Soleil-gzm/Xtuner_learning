# Copyright (c) OpenMMLab. All rights reserved.
def openai_map_fn(example):
    messages = example["messages"]
    system = ""
    input = ""
    conversation = []
    while messages and messages[0]["role"] == "assistant":
        messages = messages[1:]
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        elif msg["role"] == "user":
            input += msg["content"]
        elif msg["role"] == "assistant":
            output_with_loss = msg.get("loss", "True")
            output_with_loss = str(output_with_loss)
            output_with_loss = output_with_loss.lower() == "true"
            conversation.append(
                {
                    "system": system,
                    "input": input,
                    "output": msg["content"],
                    "output_with_loss": output_with_loss,
                }
            )
            system = ""
            input = ""
        else:
            raise NotImplementedError
    return {"conversation": conversation}

import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (
    CheckpointHook,
    DistSamplerSeedHook,
    IterTimerHook,
    LoggerHook,
    ParamSchedulerHook,
)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
from xtuner.engine.hooks import (
    DatasetInfoHook,
    EvaluateChatHook,
    VarlenAttnArgsToMessageHubHook,
)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.parallel.sequence import SequenceParallelSampler
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE


#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = "/home/zimeng/work/projects/Xtuner_learning/models/Qwen2.5-0.5B-Instruct"
use_varlen_attn = False

# Data
data_path = "datas/test_1000.jsonl"
prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 2048
pack_to_max_length = True

# parallel 
sequence_parallel_size = 1

# Scheduler & Optimizer
batch_size = 1      # 每张卡上的批大小
accumulative_counts = 16    # 梯度累积步数
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 0
max_epochs = 5      # 训练轮数。数据集小可以设大一点（如 5），数据集大就设小一点（如 1-3）。
optim_type = AdamW
lr = 2e-4       # 学习率。LoRA 微调常用 1e-4 到 2e-4。
betas = (0.9, 0.999)    # AdamW 的动量参数，一般不用改。
weight_decay = 0    # 权重衰减，LoRA 微调通常设 0。
max_norm = 1  # 梯度裁剪阈值，防止梯度爆炸。
warmup_ratio = 0.03     # 前 3% 的训练步数用于学习率预热，从极小值线性升到 lr。

# Save
save_steps = 500
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = "你是一位来自苏宁金融的电话专员，现在你需要向以下客户提醒还款。"
evaluation_inputs = ["我怎么还钱？", "How do I pay you back?"]

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side="right",
)

model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    ),
    lora=dict(
        type=LoraConfig,
        r=16,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
alpaca_en = dict(
    type=process_hf_dataset,
    dataset=dict(type='datasets.load_dataset',path='json',data_files=data_path,),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=openai_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn,
    map_num_proc=4,  # 限制为 4 个进程，避免与 20 核任务争抢 CPU
)

sampler = SequenceParallelSampler if sequence_parallel_size > 1 else DefaultSampler

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=alpaca_en,
    sampler=dict(type=sampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,       # 自动混合精度（AMP）优化器封装。FP16 计算加速，同时用动态损失缩放防止梯度下溢。
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template,
    ),
]

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # 关闭 cuDNN 自动调优，输入尺寸固定时可设 True 加速。
    cudnn_benchmark=False,      
    # 多进程配置，fork 是 Linux 默认方式。
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # 分布式后端，NCCL 是 NVIDIA GPU 的标准选择。
    dist_cfg=dict(backend="nccl"),
)

# set visualizer
visualizer = None

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
