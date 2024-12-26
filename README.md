# Survey on Efficient Large Foundation Model Inference: A Perspective From Model and System Co-Design
##### Paper Link: https://arxiv.org/abs/2409.01990

## Model Design

### Quantization
| Article Title | Year | Subfield | Link |
| --- | --- | --- | --- |
| **Quarot: Outlier-free 4-bit inference in rotated LLMs** | 2024 | Quantization | [Link](https://arxiv.org/abs/2404.00456) |
| **Sparse-quantized representation for near-lossless LLM weight compression** | 2023 | Quantization | [Link](https://arxiv.org/abs/2306.03078) |
| **GPTQ: Accurate post-training quantization for generative pre-trained transformers** | 2022 | Quantization | [Link](https://arxiv.org/abs/2210.17323) |
| **Llama.int8(): 8-bit matrix multiplication for transformers at scale** | 2022 | Quantization | [Link](https://arxiv.org/abs/2205.14135) |

### Distillation
| Article Title | Year | Subfield | Link |
| --- | --- | --- | --- |
| **On-policy distillation of language models: Learning from self-generated mistakes** | 2024 | Knowledge Distillation | [Link](https://openreview.net/forum?id=OdnNBxlShZG) |
| **Knowledge distillation for closed-source language models** | 2024 | Knowledge Distillation | [Link](https://arxiv.org/abs/2401.07013) |
| **Flamingo: A visual language model for few-shot learning** | 2022 | Knowledge Distillation | [Link](https://arxiv.org/abs/2204.14198) |
| **In-context learning distillation: Transferring few-shot learning ability of pre-trained language models** | 2022 | Knowledge Distillation | [Link](https://arxiv.org/abs/2212.10670) |
| **Less is more: Task-aware layer-wise distillation for language model compression** | 2022 | Knowledge Distillation | [Link](https://arxiv.org/abs/2210.01351) |
| **Knowledge Distillation for Closed-Source Language Models** | 2024 | Knowledge Distillation | [Link](https://openreview.net/pdf?id=oj4KUNoKnf) |
| **MiniLLM: Knowledge distillation of large language models** | 2024 | Knowledge Distillation | [Link](https://arxiv.org/pdf/2306.08543) |
| **Baby llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty** | 2023 | Knowledge Distillation | [Link](https://arxiv.org/pdf/2308.02019) |
| **Propagating knowledge updates to lms through distillation** | 2023 | Knowledge Distillation | [Link](https://arxiv.org/pdf/2306.09306) |
| **On-policy distillation of language models: Learning from self-generated mistakes** | 2024 | Knowledge Distillation | [Link](https://arxiv.org/pdf/2306.13649) |
| **Metaicl: Learning to learn in context** | 2022 | Knowledge Distillation | [Link](https://arxiv.org/pdf/2110.15943) |
| **In-context learning distillation: Transferring few-shot learning ability of pre-trained language models** | 2022 | Knowledge Distillation | [Link](https://arxiv.org/pdf/2212.10670) |
| **Multistage collaborative knowledge distillation from a large language model for semi-supervised sequence generation** | 2024 | Knowledge Distillation | [Link](https://arxiv.org/pdf/2311.08640) |

### Pruning
| Article Title | Year | Subfield | Link |
| --- | --- | --- | --- |
| **Fluctuation-based adaptive structured pruning for large language models** | 2024 | Pruning | [Link](https://arxiv.org/abs/2406.02069) |
| **SparseGPT: Massive language models can be accurately pruned in one-shot** | 2023 | Pruning | [Link](https://arxiv.org/abs/2303.12712) |
| **Pruner-zero: Evolving symbolic pruning metric from scratch for large language models** | 2024 | Pruning | [Link](https://arxiv.org/abs/2406.02924) |
| **Nash: A simple unified framework of structured pruning for accelerating encoder-decoder language models** | 2023 | Pruning | [Link](https://arxiv.org/abs/2310.10054) |

## System Design

### K-V Cache
| Article Title | Year | Subfield | Link |
| --- | --- | --- | --- |
| **PyramidKV: Dynamic KV cache compression based on pyramidal information funneling** | 2024 | KV Cache Compression | [Link](https://arxiv.org/abs/2406.02069) |
| **Quest: Query-aware sparsity for long-context transformers** | 2024 | KV Cache Compression | [Link](https://arxiv.org/abs/2401.07013) |

### Cache Eviction
| Article Title | Year | Subfield | Link |
| --- | --- | --- | --- |
| **FlashAttention: Fast and memory-efficient exact attention with I/O-awareness** | 2022 | Cache Eviction | [Link](https://arxiv.org/abs/2205.14135) |
| **Token merging: Your ViT but faster** | 2023 | Cache Eviction | [Link](http://arxiv.org/abs/2210.09461) |

### Memory Management
| Article Title | Year | Subfield | Link |
| --- | --- | --- | --- |
| **ZipVL: Efficient large vision-language models with dynamic token sparsification and KV cache compression** | 2024 | Memory Management | [Link](https://arxiv.org/abs/2410.08584) |
| **Memory use of GPT-J 6B** | 2021 | Memory Management | [Link](https://discuss.huggingface.co/t/memory-use-of-gpt-j-6b/10078) |

### MoE
| Article Title | Year | Subfield | Link |
| --- | --- | --- | --- |
| **Light-PEFT: Lightening parameter-efficient fine-tuning via early pruning** | 2024 | Fine-Tuning | [Link](https://arxiv.org/abs/2406.03792) |
| **Minillm: Knowledge distillation of large language models** | 2024 | Knowledge Distillation | [Link](https://openreview.net/forum?id=OdnNBxlShZG) |

## Model-Sys Co-Design

### Mixed Precision Training
| Article Title | Year | Subfield | Link |
| --- | --- | --- | --- |
| **Power-BERT: Accelerating BERT inference via progressive word-vector elimination** | 2020 | Mixed Precision | [Link](https://arxiv.org/abs/2005.12345) |
| **Learning both weights and connections for efficient neural network** | 2015 | Mixed Precision | [Link](https://arxiv.org/abs/1803.03635) |

### Fine-Tuning with System Optimization
| Article Title | Year | Subfield | Link |
| --- | --- | --- | --- |
| **Effective methods for improving interpretability in reinforcement learning models** | 2021 | Fine-Tuning | [Link](https://link.springer.com/article/10.1007/s10462-021-09984-x) |
| **Sparse-pruning for accelerating transformer models** | 2023 | Pruning and Optimization | [Link](https://arxiv.org/abs/2305.14135) |

