import os
import re
import torch

os.system("pip install -q transformers datasets peft bitsandbytes accelerate huggingface_hub")

from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from huggingface_hub import login

hf_token = "HALKAN_GELI_TOKEN_KAAGA"
hf_repo = "maanka2/opus-mt-synthetic-en-so-finetuned"
login(token=hf_token)

os.system("rm -rf maanka")
os.system("git clone https://github.com/user41420/maanka.git")

samples = {"source": [], "target": []}
repo_path = "maanka"
for filename in os.listdir(repo_path):
    if filename.endswith(".txt"):
        with open(os.path.join(repo_path, filename), 'r', encoding='utf-8') as f:
            content = f.read()
            blocks = re.findall(r'USER:\s*(.*?)\s*ASSISTANT:\s*(.*?)\s*(?=USER:|\Z)', content, re.DOTALL)
            for user_text, asst_text in blocks:
                if user_text and asst_text:
                    samples["source"].append(user_text.strip())
                    samples["target"].append(asst_text.strip())

dataset = Dataset.from_dict(samples)

model_id = "Helsinki-NLP/opus-mt-synthetic-en-so"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess_function(examples):
    inputs = examples["source"]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True)
    labels = tokenizer(targets, max_length=256, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)

training_args = Seq2SeqTrainingArguments(
    output_dir="./opus_mt_finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    predict_with_generate=True,
    generation_max_length=256
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    processing_class=tokenizer,
)

trainer.train()

model.push_to_hub(hf_repo)
tokenizer.push_to_hub(hf_repo)
