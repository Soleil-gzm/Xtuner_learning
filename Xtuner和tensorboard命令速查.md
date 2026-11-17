```markdown
# XTuner & TensorBoard 常用命令速查

> 环境前提：已激活 XTuner 环境（`conda activate xtuner-qwen`），且当前目录在训练工作区根目录。
> 下文用 `<config>`、`<work_dir>`、`<base_model>`、`<pth_ckpt>`、`<hf_adapter>` 表示需要替换的实际路径。

---

## 1. 训练相关

### 1.1 启动单卡训练（推荐）

```bash
xtuner train <config>.py --deepspeed deepspeed_zero2
```

### 1.2 指定训练产物保存目录

```bash
xtuner train <config>.py --deepspeed deepspeed_zero2 --work-dir <work_dir>
```

示例（你的实际命令）：

```bash
xtuner train stateful_rag_dialog_engine/train.py --deepspeed deepspeed_zero2 --work-dir stateful_rag_dialog_engine/ft/test
```

> 不指定 `--work-dir` 时，默认保存在 `./work_dirs/`。

### 1.3 列出可用官方配置

```bash
xtuner list-cfg
xtuner list-cfg | grep qwen2_5_1_5b
```

### 1.4 从 checkpoint 恢复训练

在配置文件 PART 5 中设置：

```python
load_from = "<work_dir>/iter_xxx.pth"
resume = True
```

然后重新执行 `xtuner train` 命令。

### 1.5 多卡训练（备用）

```bash
NPROC_PER_NODE=2 xtuner train <config>.py --deepspeed deepspeed_zero2
```

---

## 2. TensorBoard 监控

### 2.1 配置文件前提（PART 5）

```python
from mmengine.visualization import Visualizer, TensorboardVisBackend

visualizer = dict(
    type=Visualizer,
    vis_backends=[dict(type=TensorboardVisBackend)]
)
```

> `visualizer = None` 时不会生成 `vis_data`，TensorBoard 无数据可看。

### 2.2 启动 TensorBoard

```bash
# 指向 work_dir（推荐，自动递归扫描）
tensorboard --logdir=<work_dir>

# 指向 vis_data（更精确）
tensorboard --logdir=<work_dir>/vis_data
```

示例（你的实际路径）：

```bash
tensorboard --logdir=stateful_rag_dialog_engine/ft/test
```

### 2.3 远程服务器访问

```bash
tensorboard --logdir=<work_dir> --host 0.0.0.0 --port 6006
```

浏览器访问 `http://<服务器IP>:6006`，注意防火墙/安全组放行 6006 端口。

### 2.4 指定端口（本地冲突时）

```bash
tensorboard --logdir=<work_dir> --port 6007
```

### 2.5 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 页面空白 / 目录不存在 | 训练还没写出日志 | 等训练跑几步后刷新 |
| 没有 loss 曲线 | `visualizer = None` | 按 2.1 改成 TensorboardVisBackend |
| 找不到 `vis_data` | `--work-dir` 写错或未生效 | `ls <work_dir>` 确认实际结构 |
| 远程打不开 | 只监听 localhost | 加 `--host 0.0.0.0` 并放行端口 |

---

## 3. 训练后转换与合并

### 3.1 PTH checkpoint → HuggingFace adapter

```bash
xtuner convert pth_to_hf <config>.py <pth_ckpt> <hf_adapter>
```

### 3.2 合并 LoRA adapter 到基础模型

```bash
xtuner convert merge <base_model> <hf_adapter> <merged_output_dir>
```

> 合并后得到完整模型，可直接被 `transformers` / vLLM 加载，适合放回业务项目 `components`。

### 3.3 只保留 adapter（不合并）

业务侧加载方式：先加载 `<base_model>`，再 `PeftModel.from_pretrained(base, <hf_adapter>)`。体积小，但推理代码需要支持 PEFT。

---

## 4. 训练后对话测试

### 4.1 加载 adapter 进行对话

```bash
# Check 任务
xtuner chat <base_model> <hf_adapter> \
  --prompt-template qwen_chat \
  --system "<Check 任务的完整 System 指令>"

# Willing 任务
xtuner chat <base_model> <hf_adapter> \
  --prompt-template qwen_chat \
  --system "<Willing 任务的完整 System 指令>"
```

### 4.2 加载合并后的模型进行对话

```bash
xtuner chat <merged_output_dir> \
  --prompt-template qwen_chat \
  --system "<System 指令>"
```

### 4.3 常用参数

| 参数 | 作用 |
| --- | --- |
| `--prompt-template qwen_chat` | Qwen2.5 必须用 ChatML 模板 |
| `--system "..."` | 指定 system prompt，路由任务必需 |
| `--bits 4` / `--bits 8` | 量化加载，显存紧张时使用 |
| `--max-new-tokens 64` | 限制生成长度，路由 JSON 很短 |
| `--temperature 0.0` | 贪心解码，路由任务建议关闭随机性 |

---

## 5. 环境与依赖

### 5.1 创建环境

```bash
conda create --name xtuner-qwen python=3.10 -y
conda activate xtuner-qwen
```

### 5.2 安装 XTuner

```bash
pip install 'xtuner[all]==0.1.23'
```

> `torch 2.6.0` 与 `bitsandbytes 0.45.0` 可能不兼容，遇到报错可将 torch 降到 `2.5.1`。

### 5.3 下载基础模型（魔搭）

```python
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='<training_workspace>/models')
```

---

## 6. 目录结构参考

```text
<training_workspace>/
├── models/Qwen2.5-1.5B-Instruct/     # 基础模型
├── data/router_sft.jsonl             # 训练数据
├── configs/qwen2_5_1_5b_router_qlora.py
├── work_dirs/                        # 默认训练产物
└── merged/                           # 合并后模型

<business_project>/
└── models/qwen2_5_1_5b_router_merged/  # 训练完放回业务项目
```

---

## 7. 一键流程（最小闭环）

```bash
# 1. 启动训练
xtuner train <config>.py --deepspeed deepspeed_zero2 --work-dir <work_dir>

# 2. 另开终端看曲线
tensorboard --logdir=<work_dir> --host 0.0.0.0 --port 6006

# 3. 训练结束，转 HF adapter
xtuner convert pth_to_hf <config>.py <work_dir>/iter_xxx.pth <hf_adapter>

# 4. 合并
xtuner convert merge <base_model> <hf_adapter> <merged_output_dir>

# 5. 交互测试两种 prompt
xtuner chat <merged_output_dir> --prompt-template qwen_chat --system "<Check System>"
xtuner chat <merged_output_dir> --prompt-template qwen_chat --system "<Willing System>"
```

---

## 8. 训练配置关键字段速查

| 字段 | 位置 | 说明 |
| --- | --- | --- |
| `pretrained_model_name_or_path` | PART 1 / PART 2 | 基础模型路径 |
| `prompt_template` | PART 1 | Qwen2.5 用 `PROMPT_TEMPLATE.qwen_chat` |
| `data_path` | PART 1 | 训练 jsonl 路径 |
| `max_length` | PART 1 | 路由任务 1024~2048 足够 |
| `max_epochs` | PART 1 | 建议 3~5 起步 |
| `batch_size` / `accumulative_counts` | PART 1 | 1.5B QLoRA 单卡可 1/8~16 |
| `lr` | PART 1 | LoRA 常用 `2e-4`，不稳定降到 `1e-4` |
| `dataset_map_fn` | PART 3 | 必须换成自定义 `router_map_fn` |
| `lora.r` | PART 2 | 1.5B 从 16 或 32 起步，64 偏大 |
| `quantization_config` | PART 2 | 4-bit QLoRA 对 1.5B 足够 |
| `visualizer` | PART 5 | 改成 TensorboardVisBackend |
| `SYSTEM` / `evaluation_inputs` | PART 5 | 仅影响训练中抽样展示，不影响训练 |

```
