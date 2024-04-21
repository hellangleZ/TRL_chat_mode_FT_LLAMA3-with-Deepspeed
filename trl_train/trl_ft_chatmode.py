#导入需要的库文件
import torch
from accelerate import PartialState
from trl import SFTTrainer,setup_chat_format
from random import randrange
from datasets import  load_dataset
from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)


#并行
device_string = PartialState().process_index
device_map={'':device_string}

# Load dataset from the hub or local fold
dataset = load_dataset("silk-road/Wizard-LM-Chinese-instruct-evol", split="train")

# Convert dataset to messages which use Openai format
system_message = """You are a LLM assistant. Users will ask you questions in Chinese, You will answer questions in Chinese :
{schema}"""

def create_conversation(sample):
      return {
        "messages": [
          {"role": "system", "content": system_message},#.format(schema=sample["context"])},
          {"role": "user", "content": sample["instruction_zh"]},
          {"role": "assistant", "content": sample["output_zh"]}
        ]
      }

#radom shuffle and select 60000 samples

dataset = dataset.shuffle().select(range(60000))

# Convert dataset to OAI messages
dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)
# split dataset into 60,000 training samples and 2000 test samples，total 70000 samples
dataset = dataset.train_test_split(test_size=2000/60000)

#show data samples demo
#print(dataset["train"][345]["messages"])

# save datasets （train/test）to disk
dataset["train"].to_json("train_dataset.json", orient="records")
dataset["test"].to_json("test_dataset.json", orient="records")



# Load train jsonl data from disk
dataset = load_dataset("json", data_files="train_dataset.json", split="train")




# Hugging Face model id or Local fold
model_id = "/data2/lm3-chat_2024_04_21" 

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

#tp
model.config.pretraining_tp = 2

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right' # to prevent warnings

# # set chat template to OpenAI chatmode, remove if you start from a fine-tuned model
model, tokenizer = setup_chat_format(model, tokenizer)


# LoRA config need more specific so change number to up for the lora_alpha and r
peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)

# Hyper-config for training
args = TrainingArguments(
    output_dir="/data2/lama-3-chat_2nd", # directory to save and repository id
    num_train_epochs=2,                     # number of training epochs
    per_device_train_batch_size=2,          # batch size per device during training
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    gradient_checkpointing_kwargs={'use_reentrant':False},
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=1,                       # log every 1 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=1e-5,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    #push_to_hub=True,                       # push model to hub if you want
    report_to="tensorboard",                # report metrics to tensorboard
)

max_seq_length = 8192 # max sequence length for model and packing of the dataset

#Train
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)

trainer.train()



