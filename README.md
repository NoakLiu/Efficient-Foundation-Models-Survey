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
| **Llm-qat: Data-free quantization aware training for large language models** | 2023 | Quantization | [Link](https://arxiv.org/abs/2305.17888) |
| **Bitdistiller: Unleashing the potential of sub-4-bit llms via self-distillation** | 2024 | Quantization | [Link](https://arxiv.org/abs/2402.10631) |
| **OneBit: Towards Extremely Low-bit Large Language Models** | 2024 | Quantization | [Link](https://arxiv.org/abs/2402.11295) |
| **QLoRA: efficient finetuning of quantized LLMs** | 2023 | Quantization | [Link](https://arxiv.org/abs/2305.14314) |
| **Memory-efficient fine-tuning of compressed large language models via sub-4-bit integer quantization** | 2023 | Quantization | [Link](https://arxiv.org/abs/2305.14152) |
| **Loftq: Lora-fine-tuning-aware quantization for large language models** | 2023 | Quantization | [Link](https://arxiv.org/abs/2310.08659) |
| **L4q: Parameter efficient quantization-aware training on large language models via lora-wise lsq** | 2024 | Quantization | [Link](https://arxiv.org/abs/2402.04902) |
| **Qa-lora: Quantization-aware low-rank adaptation of large language models** | 2023 | Quantization | [Link](https://arxiv.org/abs/2309.14717) |
| **Efficientqat: Efficient quantization-aware training for large language models** | 2024 | Quantization | [Link](https://arxiv.org/abs/2407.11062) |
| **Accurate lora-finetuning quantization of llms via information retention** | 2024 | Quantization | [Link](https://arxiv.org/abs/2402.05445) |
| **EdgeQAT: Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge** | 2024 | Quantization | [Link](https://arxiv.org/abs/2402.10787) |

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
| **Minillm: Knowledge distillation of large language models** | 2024 | Knowledge Distillation | [Link](https://openreview.net/forum?id=OdnNBxlShZG) |

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
| **ZipVL: Efficient Large Vision-Language Models with Dynamic Token Sparsification and KV Cache Compression** | 2024 | KV Cache Compression | [Link](https://arxiv.org/abs/2410.08584) |
| **VL-Cache: Sparsity and Modality-Aware KV Cache Compression for Vision-Language Model Inference Acceleration** | 2024 | KV Cache Compression | [Link](https://arxiv.org/abs/2410.23317) |

## Token Sparsification

| Article Title | Year | Subfield | Link |
| --- | --- | --- | --- |
| **DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification** | 2021 | Dynamic Token Sparsification | [Link](https://arxiv.org/abs/2104.04482) |
| **A-ViT: Adaptive Tokens for Efficient Vision Transformer** | 2022 | Adaptive Tokens | [Link](https://arxiv.org/abs/2206.02696) |
| **Quest: Query-Aware Sparsity for Long-Context Transformers** | 2024 | Query-Aware Sparsity | [Link](https://arxiv.org/abs/2401.07013) |
| **Token-Level Transformer Pruning** | 2022 | Token Pruning | [Link](https://arxiv.org/abs/2203.03229) |
| **Token Merging: Your ViT but Faster** | 2023 | Token Merging | [Link](http://arxiv.org/abs/2210.09461) |


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
| **An Evolved Universal Transformer Memory** | 2024 | Neural Attention Memory Models | [Link](https://arxiv.org/abs/2410.13166) |
| **Efficient Memory Management for Large Language Model Serving with PagedAttention** | 2023 | PagedAttention | [Link](https://arxiv.org/abs/2309.06180) |
| **Recurrent Memory Transformer** | 2023 | Recurrent Memory, Long Context Processing | [Link](https://github.com/booydar/recurrent-memory-transformer) |
| **SwiftTransformer: High Performance Transformer Implementation in C++** | 2023 | Model Parallelism, FlashAttention, PagedAttention | [Link](https://github.com/LLMServe/SwiftTransformer) |
| **ChronoFormer: Memory-Efficient Transformer for Time Series Analysis** | 2024 | Memory Efficiency, Time Series Data | [Link](https://github.com/kyegomez/ChronoFormer) |


### MoE
| Article Title | Year | Subfield | Link |
| --- | --- | --- | --- |
| **Mixtral of Experts** | 2024 | Sparse MoE, Routing Networks | [Link](https://arxiv.org/abs/2401.04088) |
| **Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts** | 2024 | Time Series, Large-Scale Models | [Link](https://arxiv.org/abs/2409.16040) |
| **AdaMV-MoE: Adaptive Multi-Task Vision Mixture-of-Experts** | 2024 | Vision, Multi-Task Learning | [Link](https://paperswithcode.com/paper/adamv-moe-adaptive-multi-task-vision-mixture) |
| **DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding** | 2024 | Vision-Language, Multimodal Models | [Link](https://arxiv.org/abs/2412.10302) |
| **Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts** | 2024 | Multimodal LLMs, Unified Models | [Link](https://paperswithcode.com/paper/uni-moe-scaling-unified-multimodal-llms-with) |

## Retrieval-Augmented Models

| Article Title | Year | Subfield | Link |
| --- | --- | --- | --- |
| **REALM: Retrieval-Augmented Language Model Pre-Training** | 2020 | Retrieval-Augmented Models | [Link](https://arxiv.org/abs/2002.08909) |
| **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** | 2020 | Retrieval-Augmented Models | [Link](https://arxiv.org/abs/2005.11401) |

## Transformer Architecture Improvements

| Article Title | Year | Subfield | Link |
| --- | --- | --- | --- |
| **Roformer: Enhanced Transformer with Rotary Position Embedding** | 2021 | Positional Embedding | [Link](https://arxiv.org/abs/2104.09864) |
| **Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned** | 2019 | Self-Attention Analysis | [Link](https://arxiv.org/abs/1904.02813) |
| **Sixteen Heads Are Better Than One: Attention Mechanisms in Transformers** | 2019 | Multi-Head Attention | [Link](https://arxiv.org/abs/1907.00263) |


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
| **Efficient Fine-Tuning and Compression of Large Language Models with Low-Rank Adaptation and Pruning** | 2023 | Fine-Tuning and Compression | [Link](https://arxiv.org/abs/2302.08582) |
| **Layer-wise Knowledge Distillation for BERT Fine-Tuning** | 2022 | Knowledge Distillation | [Link](https://arxiv.org/abs/2204.05155) |
| **Light-PEFT: Lightening parameter-efficient fine-tuning via early pruning** | 2024 | Fine-Tuning | [Link](https://arxiv.org/abs/2406.03792) |

