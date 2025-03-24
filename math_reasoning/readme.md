# Math Reasoning Fine-tuning Project

## Overview
This project demonstrates the effectiveness of supervised fine-tuning (SFT) on improving mathematical reasoning capabilities in large language models. We use the Llama-3.2-3B-Instruct model as our base and fine-tune it on a math-specific dataset.

## Model Details
**Base Model**:  
- `unsloth/Llama-3.2-3B-Instruct`

**Fine-tuned Model**:  
- `arpansm10/Llama-3.2-3B-Math-Reasoning`
- Trained with LoRA (Low-Rank Adaptation)
- 4-bit QLora quantization for efficient training
- Sequence length: 4096 tokens

## Dataset Used
**Training Data**:  
- OpenThoughts-114k-math dataset
- 5,000 randomly selected math reasoning examples
- Filtered for sequence length ≤ 4096 tokens

**Evaluation Data**:  
- GSM-Hard benchmark (100 samples)
- Grade school math problems with increased difficulty
- Contains multi-step reasoning challenges

## Key Libraries
- **unsloth**: For accelerated training/inference
- **PEFT**: Parameter-Efficient Fine-Tuning
- **TRL**: Transformer Reinforcement Learning
- **litellm**: Unified LLM API interface
- **datasets**: Hugging Face dataset management

## Results
| Model | Dataset | Accuracy | Improvement |
|-------|---------|----------|-------------|
| Llama-3.2-3B-Instruct (Base) | GSM-Hard (100 samples) | 23% | - |
| Llama-3.2-3B-Math-Reasoning (Fine-tuned) | GSM-Hard (100 samples) | 46% | +100% |

Key findings:
- 100% relative improvement in accuracy after fine-tuning
- Demonstrates effectiveness of domain-specific adaptation
- Achieved using parameter-efficient fine-tuning (PEFT) with LoRA

## Project Structure
