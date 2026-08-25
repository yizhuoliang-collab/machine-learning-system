> 版本 v0.1（AI 初稿）· 使用提示词 lab_readme

# L07 LoRA 参数高效微调实验

**实验编号**：L07　　**建议学时**：2–3 小时　　**对应内容**：LoRA 与监督微调

## 实验简介

本实验使用 `ms-swift` 在华为云 ModelArts 的 Ascend NPU 上，对 `Qwen/Qwen3-0.6B` 做 LoRA 监督微调（SFT）。实验从 LoRA 原理和环境准备开始，先用 3 个 step 跑通训练链路，再用 50 个 step 观察适配器更新和 loss 曲线。

| Notebook | 内容 | 训练步数 |
| --- | --- | ---: |
| `notebooks/L07-01_lab_introduction.ipynb` | LoRA 原理、术语、参数和环境准备 | 0 |
| `notebooks/L07-02_ms_swift_smoke_test.ipynb` | 读取 SFT 数据，生成并执行 LoRA 训练命令 | 3 |
| `notebooks/L07-03_ms_swift_train_and_loss_analysis.ipynb` | 完整 LoRA 训练、日志解析和 loss 曲线 | 50 |

三个 notebook 按顺序从头运行。每个 notebook 都包含自己的初始化代码，运行前请确认 JupyterLab 使用 `Python (PyTorch-2.7.1)` kernel。

## 一、实验任务

### 学习目标

完成本实验后，你应当能：

1. 写出 LoRA 的低秩更新公式，解释基础模型、适配器、`rank`、`alpha` 和目标层。
2. 读懂 `swift sft` 的 LoRA 参数，说明哪些参数被冻结、哪些参数参与更新。
3. 在 ModelArts 上完成数据准备、smoke test 和 50-step 正式训练。
4. 从 `logging.jsonl` 绘制 loss 曲线，解释 raw loss 和 moving average 的区别。

### LoRA 公式

全参数微调直接更新线性层权重 `W`。LoRA 冻结基础权重，只学习一个低秩增量：

```text
W' = W + ΔW
ΔW = (alpha / r) × B × A
```

`r` 是低秩分支的维度，`A` 和 `B` 是新增的可训练矩阵，`alpha` 控制增量的缩放。对于形状为 `d_out × d_in` 的线性层，全参数更新需要 `d_out × d_in` 个参数，LoRA 分支只需要 `r × (d_in + d_out)` 个参数。

## 二、任务准备

### 前置知识

- Transformer 的线性层、注意力和 token：对应课程大模型基础内容。
- 监督微调（SFT）和对话数据格式：对应 L07 实验导学。
- Python、JSONL 和 JupyterLab 的基本使用。

### ModelArts 创建配置

在 ModelArts 控制台创建 Notebook 时按下表选择。控制台字段名称可能随版本变化，以实际页面为准。

| 配置项 | 本实验配置 |
| --- | --- |
| 区域 | 西南-贵阳一 |
| 资源池 | 公共资源池 |
| 实例规格 | `1 × ascend-snt9b3`，24 vCPUs，192 GiB |
| 镜像 | `pytorch_ascend: pytorch_2.7.1-cann_8.5.2-py_3.12-hce_2.0.2512-aarch64-snt9b` |
| WebIDE | JupyterLab 4.4.10 |
| 系统盘 | 50 GiB |
| 数据盘 | EVS 50 GiB，挂载到 `/home/ma-user/work` |
| 网络 | 公共网络，允许访问公网，访问方式选公有网关 |
| 自动停止 | 建议开启；训练时间较长时可按需要设置 4–10 小时 |
| SSH 远程开发 | 需要使用本地 SSH 上传或执行时开启，并选择已创建的密钥对 |

创建完成后等待实例状态变为“运行中”，进入 JupyterLab。右上角 kernel 选择：

```text
Python (PyTorch-2.7.1)
```

这个 kernel 需要与镜像中的 PyTorch、`torch_npu` 和 CANN 运行时配套。不要在错误的 Python kernel 中安装或运行训练代码。

### 目录结构

```text
.
├── notebooks/
│   ├── L07-01_lab_introduction.ipynb
│   ├── L07-02_ms_swift_smoke_test.ipynb
│   └── L07-03_ms_swift_train_and_loss_analysis.ipynb
└── 07_labs/lab_07_lora_finetune/
    └── README.md
```

运行产物默认写入 notebook 工作目录下的 `L07-0x_workspace/`，包括模型缓存、`train.jsonl`、日志、曲线和 checkpoint。重要文件放在 `/home/ma-user/work` 下，实例停止后仍可保留。

## 三、任务实施

### 步骤 1：L07-01 导学与环境准备

打开 `L07-01_lab_introduction.ipynb`，从第一个 code cell 开始运行。

本节会：

- 检查 Python、PyTorch、`torch_npu` 和 NPU 是否匹配；
- 安装缺失的 `modelscope`、`ms-swift` 和 `matplotlib`；
- 下载 `Qwen/Qwen3-0.6B`；
- 生成 JSONL 对话数据；
- 讲解 LoRA 公式和训练参数。

完成标志是输出中出现模型路径、训练数据路径和 NPU 数量。

### 步骤 2：L07-02 smoke test

打开 `L07-02_ms_swift_smoke_test.ipynb`，从头运行到最后。

重点看训练命令中的参数：

```text
--tuner_type lora
--target_modules all-linear
--lora_rank 8
--lora_alpha 32
--learning_rate 1e-4
--max_steps 3
```

本节会运行 3 个 step，检查数据能否读取、NPU 能否完成训练、日志是否产生以及 checkpoint 是否保存。输出中应能看到有限的 loss 记录和 checkpoint 路径。

### 步骤 3：L07-03 正式训练

打开 `L07-03_ms_swift_train_and_loss_analysis.ipynb`，从头运行到最后。

正式配置为：

| 参数 | 值 |
| --- | ---: |
| 训练记录 | 54 条（6 条教学样本重复 9 次） |
| `max_steps` | 50 |
| `logging_steps` | 1 |
| `save_steps` | 10 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 1 |
| `max_length` | 512 |

训练完成后，代码会从原始 `logging.jsonl` 读取逐 step 的 `loss`，绘制 raw loss 和 5-step moving average，并输出 checkpoint 和曲线路径。raw loss 有波动是正常现象，先看 moving average 的整体趋势，再结合训练参数解释曲线。

### 结果检查

应能找到以下文件：

```text
L07-03_workspace/
├── train.jsonl
├── environment.json
└── L07-03_train/
    ├── notebook_stdout.log
    ├── loss_curve.png
    ├── logging.jsonl
    └── v0-*/checkpoint-50/
```

### 运行异常时

- JupyterLab 提示“磁盘内容已改变”：先确认当前 notebook 是否被其他窗口或 SSH 上传覆盖，再刷新页面并选择最新文件。
- 找不到 NPU：检查实例规格和 kernel 是否为 `Python (PyTorch-2.7.1)`。
- 模型下载失败：检查实例是否允许访问公网，并确认 `/home/ma-user/work` 有足够空间。
- OOM：先降低 `max_length`，再考虑减小 batch 或 LoRA 目标层范围。
- loss 出现波动：查看完整曲线和日志，不要只根据一个 step 判断训练走势。

## 四、实验报告

报告至少包含以下内容：

1. **原理**：写出 `ΔW = (alpha / r) × B × A`，说明本实验的 `rank=8`、`alpha=32` 和 `all-linear`。
2. **配置**：记录 ModelArts 规格、镜像、kernel、模型、数据条数、batch、学习率和训练步数。
3. **过程**：说明 L07-02 是否跑通，L07-03 是否到达 `50/50`，附 `logging.jsonl` 和 checkpoint 路径。
4. **曲线**：提交 `loss_curve.png`，描述 raw loss 的波动和 moving average 的变化。
5. **思考**：讨论把 `rank` 改为 16，或只训练 `q_proj`、`v_proj` 后，适配器参数量和训练结果可能如何变化。

本实验的 notebook 没有 TODO，不设置 `answer/` 文件夹。

## 五、实验总结

本实验把 LoRA 的低秩更新公式落到一个可运行的 SFT 流程中：先准备基础模型和对话数据，再用 smoke test 检查命令，最后完成 50-step 训练并读取日志。报告中应能说明“哪些参数在更新、为什么这样配置、曲线反映了什么”。

## 提交清单

- [ ] 三个 notebook 按顺序运行完成。
- [ ] L07-02 产生有限 loss 和 checkpoint。
- [ ] L07-03 到达 50 step，并生成 `loss_curve.png`。
- [ ] 报告包含 LoRA 公式、配置、日志路径和曲线分析。
- [ ] 保留原始日志，不只提交截图。

常见问题可在实验运行记录中补充；当前资源包不包含 `faq.md`、`check_env.sh`、`run.sh` 和 `check_result.py`，后续如课程统一增加检查脚本，再补充对应入口。
