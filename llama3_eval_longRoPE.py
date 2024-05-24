from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from longrope import long_rope, LongRoPEWrapper

model_id = "Meta/Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# Define LongRoPE scaling factors ( without fine-tune sinario, use 8x)
dim = model.config.hidden_size
lambda_factors_8x = torch.linspace(1.0, 1.8, dim // 2)
lambda_factors_16x = torch.linspace(1.0, 2.5, dim // 2)
n_prime = 128  # adjust value

# Choose scale factor
use_8x = True  # Set this to False to use 16x scaling
lambda_factors = lambda_factors_8x if use_8x else lambda_factors_16x

# Wrap the base model with LongRoPE
model_new = LongRoPEWrapper(model, lambda_factors=lambda_factors, n_prime=n_prime)



terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

messages = [
    {"role": "system", "content": "You are a helpful AI assist to answer any user's input . answer would be long as possible to include the detail knowledge."},
    {"role": "user", "content": "could you explain about compressive transformer compare to transformerXL and legacy transformer with python code ?"},
]


input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

import time 
num_try = 10
for i in range(num_try): 
    print( f"try {i}/{num_try} ")
    tic = time.time()
    outputs = model_new.generate(
        input_ids,
        max_new_tokens=7000,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    toc = time.time()
    dur = toc - tic 
    response = outputs[0][input_ids.shape[-1]:]
    output_text = tokenizer.decode(response, skip_special_tokens=True)
    print(f"{len(output_text)} with {dur:4.2f}sec " )
    print(output_text)
    print("-----------------------")

