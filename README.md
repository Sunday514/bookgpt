# BookGPT: 本地中文写作风格微调流水线

BookGPT 用来把一份你有合法使用权的中文 `.txt` 文本，转换成可训练数据，并在本地用 `Unsloth QLoRA` 做中文写作风格微调。

当前默认的数据构建、提示模板和训练配置，面向的是：

- 中文创意写作
- 中文小说、散文、书信、对白等中文作品片段续写或改写
- 中文风格对齐

不面向：

- 英文写作微调
- 通用文档问答
- 多语言混合训练

## 项目能力

当前主流程支持：

1. 输入任意中文 `.txt`
2. 解析目录、章节、正文段落
3. 构建以 `continue` 为主的中文写作训练样本
4. 混入外部中文指令/对话数据，保留基础能力
5. 按 tokenizer 长度过滤样本
6. 对语言模型做本地 LoRA 微调

## 默认假设

- 训练语言：中文
- 训练框架：`Unsloth`
- 支持范围：原则上支持 `Unsloth` 当前支持的语言模型
- 默认训练长度：`2048`
- 默认训练步数：`320`
- 默认真实 batch size：`32`
- 默认训练方式：只保存 checkpoint，不做中途 eval

当前默认配置已经在单卡约 `48GB` 显存环境下验证可运行。

## 仓库不包含的内容

以下内容不随仓库分发，需要你自己下载或生成：

- 模型权重
- Hugging Face 外部数据集
- 归一化后的外部数据
- 生成的训练集
- 训练输出

本仓库只提交：

- 代码
- 配置
- 文档
- 一个很小的中文示例文本

## 目录结构

```text
bookgpt/
├── bookgpt/
│   ├── common.py
│   └── data/
├── configs/
│   └── train.yaml
├── dataset/
│   ├── raw_books/
│   ├── book_runs/
│   ├── final_runs/
│   ├── external_normalized/
│   ├── README.md
│   └── schema.md
├── eval/
├── scripts/
├── serve/
├── models/                # 本地下载，不提交
├── external_datasets/     # 本地下载，不提交
└── outputs/               # 本地生成，不提交
```

详见 [dataset/README.md](dataset/README.md) 和 [scripts/README.md](scripts/README.md)。

## 上游模型与数据

### 示例模型

下面这个模型是当前仓库默认验证过的示例：

- `unsloth/Qwen2.5-32B-Instruct-bnb-4bit`
  - <https://huggingface.co/unsloth/Qwen2.5-32B-Instruct-bnb-4bit>

### 示例数据集

- `wenbopan/OpenHermes-2.5-zh`
  - <https://huggingface.co/datasets/wenbopan/OpenHermes-2.5-zh>
  - `Apache-2.0`
- `OpenAssistant/oasst1`
  - <https://huggingface.co/datasets/OpenAssistant/oasst1>
  - `Apache-2.0`
- `m-a-p/COIG-Writer`
  - <https://huggingface.co/datasets/m-a-p/COIG-Writer>
  - `ODC-BY`

你需要自行确认这些上游资源的许可证和使用条件。

## 许可证

本仓库代码采用 [MIT License](LICENSE)。

注意：上游模型、数据集以及你自己的文本材料，不受本仓库 MIT 许可证覆盖，仍然遵循各自原始条款。

## 环境准备

### 1. 创建环境

```bash
conda create -p ./env python=3.11 -y
conda activate ./env
```

### 2. 安装 Unsloth

按官方说明安装：

- <https://github.com/unslothai/unsloth>

### 3. 安装其余依赖

```bash
pip install pyyaml datasets pyarrow tqdm
```

## 下载模型与数据

使用 `hf-mirror`：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

下载示例模型：

```bash
hf download unsloth/Qwen2.5-32B-Instruct-bnb-4bit \
  --local-dir models/Qwen2.5-32B-Instruct-bnb-4bit
```

下载示例外部数据集：

```bash
hf download --repo-type dataset wenbopan/OpenHermes-2.5-zh \
  --local-dir external_datasets/OpenHermes-2.5-zh

hf download --repo-type dataset OpenAssistant/oasst1 \
  --local-dir external_datasets/oasst1

hf download --repo-type dataset m-a-p/COIG-Writer \
  --local-dir external_datasets/COIG-Writer
```

归一化外部数据：

```bash
python scripts/prepare_external_datasets.py \
  --output-dir dataset/external_normalized
```

## 快速验证

仓库自带一个中文示例文本：

- `dataset/raw_books/example_book.txt`

运行：

```bash
python scripts/run_smoke_pipeline.py
```

## 正常使用流程

把你自己的中文 `.txt` 放到 `dataset/raw_books/`，然后执行：

```bash
python scripts/run_pipeline.py \
  --book-input dataset/raw_books/your_text.txt
```

这一步会生成：

- `dataset/book_runs/current/`
- `dataset/final_runs/current/`

然后开始训练：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UNSLOTH_DISABLE_STATISTICS=1
python scripts/train_unsloth.py --config configs/train.yaml
```

## 检查数据集

```bash
python scripts/inspect_dataset.py \
  --dataset-dir dataset/final_runs/current \
  --tokenizer-path models/Qwen2.5-32B-Instruct-bnb-4bit \
  --write-summary
```

## 推理与评估

离线生成评估：

```bash
python scripts/eval_model.py \
  --model-path outputs/train_2048/checkpoint-240 \
  --prompts eval/prompts.jsonl \
  --output eval/results.jsonl
```

本地聊天：

```bash
python scripts/chat_local.py \
  --model-path outputs/train_2048/checkpoint-240
```

如果你要比较不同训练阶段，可以把 `--model-path` 换成其他 `checkpoint-*` 目录。

## 当前状态

当前项目已经是一套可运行的 MVP，可以完成：

- 中文 `.txt` -> 训练集构建
- 本地 QLoRA 微调
- 训练后本地推理验证

当前主要工作重点已经不是“流程能不能跑通”，而是“风格效果是否达到预期”。

## 注意事项

- 你只能使用自己有合法使用权的中文文本。
- 仓库不会提供受版权保护的原始书籍。
- `bookgpt` 这个名字可以用，但建议公开发布时配合更明确的副标题。
