import os


import torch
from transformers import AutoTokenizer, AutoModel
import loralib as lora
from peft import get_peft_model, LoraConfig, TaskType

device = 'cuda'
checkpoint = "THUDM/chatglm-6b"
model_id = "finetune_test"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, revision = 'main')
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, revision = 'main')


mixed_precision = 'bf16'
lora_config = LoraConfig(
    r = 4,
    lora_alpha = 4,
    lora_dropout = 0.05,
    #enable_lora= [True, False, True],
)

LR = 1e-4
BATCH = 1
MAX_LENGTH = 256
NUM_EPOCHS = 3
accumulate_step = 8
warm_up_ratio = 0.1

epoch = NUM_EPOCHS - 1
model = get_peft_model(model, lora_config)

model.load_state_dict(torch.load(f'saved/{model_id}/{model_id}_epoch_{epoch}.pt'), strict=False)

model.half().cuda().eval()
history = []

input_format =  'Instruction: What is the sentiment of this news?\nAnswer:{very negative/negative/neutral/positive/very positive}\nInput:"%s"\nAnswer:'

while True:
    content = input(">")
    response, history = model.chat(tokenizer, input_format%content, history=history)
    print(response)