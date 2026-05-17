# Light-MER: Multimodal Emotion Understanding with Knowledge Distillation

> **Note**: This repository accompanies a paper currently under peer review. Certain core algorithmic implementations (Optimal Transport distillation loss, GRPO reward functions, and teacher model configuration) have been withheld to protect intellectual property prior to publication. The full implementation will be released upon paper acceptance.

## Overview

Light-MER is a multimodal large language model (MLLM) for emotion understanding from video, audio, and text. This extended version introduces:

- **Sliced Wasserstein Distance (SWD) Distillation**: Compresses an 8B-parameter teacher into a <1B student model with minimal performance loss, using optimal transport on hidden-state representations.
- **GRPO Fine-tuning**: Applies Group Relative Policy Optimization to improve output format consistency and emotion label accuracy without supervised labels.
- **MER-UniBench Evaluation**: Comprehensive evaluation across 9 multimodal emotion recognition benchmarks.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Light-MER (Student)         │
                    │                                     │
  Video Frames ──► │  CLIP-ViT ──► Q-Former ──┐         │
                    │                          ├──► Qwen3-0.6B ──► Caption
  Audio Wav ────► │  HuBERT ────► Q-Former ──┘         │
                    │                                     │
                    └─────────────────────────────────────┘
                              ▲
                              │ SWD Distillation (OT alignment on hidden states)
                              ▼
                    ┌─────────────────────────────────────┐
                    │         Teacher (Frozen)            │
                    │  CLIP-ViT-L + HuBERT-L + Qwen3-8B  │
                    └─────────────────────────────────────┘
```

## Project Structure

```
Light-MER/
├── train.py                        # Training entry point (with OT distillation)
├── evaluation.py                   # Full evaluation on MER-UniBench
├── inference_sample.py             # Single-video inference
├── grpo_reward.py                  # GRPO reward functions [redacted]
├── config.py                       # Global paths and constants
├── environment.yml                 # Conda environment
│
├── my_affectgpt/                   # Core library
│   ├── models/
│   │   ├── affectgpt.py            # Main model (multimodal LLM + OT hooks)
│   │   ├── ot_loss.py              # OT/Sinkhorn loss [partially redacted]
│   │   ├── encoder.py              # Visual & acoustic encoder registry
│   │   ├── Qformer.py              # Query-Former for cross-modal alignment
│   │   ├── blip2.py                # BLIP-2 base class
│   │   └── ImageBind/              # ImageBind encoder (optional)
│   ├── datasets/
│   │   ├── builders/               # Dataset builder registry
│   │   └── datasets/               # Per-dataset implementations (MER, MELD, IEMOCAP, SIMS, ...)
│   ├── evaluation/                 # Metrics (emotion wheel, weighted F1, etc.)
│   ├── processors/                 # Video/image/text preprocessing
│   ├── runners/                    # Distributed training loop
│   ├── tasks/                      # Task abstraction layer
│   └── common/                     # Registry, config, distributed utils, optimizers
│
├── toolkit/                        # Auxiliary utilities
│   ├── dataloader/                 # Legacy data loading
│   ├── preprocess/                 # Dataset preprocessing scripts
│   └── utils/                      # GPT scoring, file I/O, Qwen helpers
│
└── train_configs/                  # YAML experiment configurations
    ├── v8_swd_hidden.yaml          # SWD on hidden states
    ├── v11_swd_logits.yaml         # SWD on output logits
    └── ...
```

## Key Technical Contributions

### 1. Optimal Transport Distillation (SWD)
- Projects teacher (4096-dim) and student (1024-dim) hidden states into a shared space
- Aligns distributions via Sinkhorn divergence with cosine cost
- Frozen teacher projection prevents "shortcut" collapse
- CE-aligned masking focuses distillation on answer tokens only

### 2. GRPO Reward Engineering
- Format reward: ensures output follows "The character's emotional state is X, Y, Z."
- Brevity reward: penalizes excessive reasoning after the emotion label
- Emotion overlap reward: measures alignment with ground truth labels

### 3. Evaluation Pipeline
- Supports 9 benchmarks: MER2023, MER2024, MELD, IEMOCAP, SIMS, SIMSv2, CMU-MOSI, CMU-MOSEI, OV-MERD+
- Metrics: WAR/UAR (classification), WAF1 (weighted F1), emotion wheel scoring
- LLM-as-judge evaluation for open-ended emotion captioning

## Requirements

- Python 3.10
- PyTorch 2.4.0 (CUDA 12.1)
- Transformers 4.49.0
- vLLM 0.6.1 (for fast inference)

```bash
conda env create -f environment.yml
```

## Usage

### Training (with OT Distillation)
```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py --cfg-path=train_configs/v8_swd_hidden.yaml
```

### Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py
```

### Single Video Inference
```bash
CUDA_VISIBLE_DEVICES=0 python -u inference_sample.py --zeroshot \
    --video_path='demo/sample.mp4' \
    --audio_path='demo/sample.wav' \
    --subtitle="I don't know what to say." \
    --cfg-path=train_configs/v8_swd_hidden.yaml \
    --options "inference.test_epoch=30"
```

## Code Availability Notice

The following components are **redacted** in this repository due to the paper being under review:

| File | Redacted Content | Reason |
|------|-----------------|--------|
| `my_affectgpt/models/ot_loss.py` | `sinkhorn_divergence()`, `OTProjector.forward()` | Core novel loss function |
| `my_affectgpt/models/affectgpt.py` | `set_teacher()` body, OT loss computation block | Novel distillation integration |
| `train.py` | `build_teacher_model()` implementation | Teacher setup specifics |
| `grpo_reward.py` | All reward function bodies | Novel reward engineering |

The full implementation will be open-sourced upon publication. Function signatures, docstrings, and architectural design are preserved to demonstrate the engineering approach.

## License

Apache 2.0 - Non-commercial research use only.
