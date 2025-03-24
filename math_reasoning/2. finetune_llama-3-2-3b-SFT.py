# install dependencies
# # Do this only in Colab notebooks! Otherwise use pip install unsloth
# !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo datasets litellm
# !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
# !pip install --no-deps unsloth

from unsloth import FastLanguageModel
import torch
import os
# load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)

# get peft model
model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 8,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# get chat template
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer,chat_template = "llama-3.1")

# formatting prompts function
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return {"text" : texts}

# load dataset
from datasets import Dataset
from datasets import load_dataset
finetune_ds = load_dataset("open-r1/OpenThoughts-114k-math")

# select 5000 examples
finetune_ds = finetune_ds['train'].select(range(5000))

# Filter out examples with token length > 4096
filtered_ds = []
for example in finetune_ds:
    if len(tokenizer(str(example['conversations']))[0]) <= 4096:
        filtered_ds.append(example)

print(f"Original dataset size: {len(finetune_ds)}")
print(f"Filtered dataset size: {len(filtered_ds)}")
filtered_ds = Dataset.from_list(filtered_ds)

# standardize sharegpt
from unsloth.chat_templates import standardize_sharegpt
finetune_ds = standardize_sharegpt(filtered_ds)
finetune_ds = finetune_ds.map(formatting_prompts_func, batched = True)

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

# train model
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = finetune_ds,
    dataset_text_field = "text",
    max_seq_length = 4096,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc

    ),
)

# train on responses only
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# tokenizer
space = tokenizer(" ", add_special_tokens = False).input_ids[0]
tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])

# train
trainer_stats = trainer.train()

# push to hub
hf_token = os.getenv("HF_TOKEN")
model.push_to_hub("arpansm10/Llama-3.2-3B-Math-Reasoning", token = hf_token)
tokenizer.push_to_hub("arpansm10/Llama-3.2-3B-Math-Reasoning", token = hf_token)

