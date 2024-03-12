from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

local_model_path = "F:\Gemma\gemma-7b-it"

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto", torch_dtype=torch.float16)

input_text = "Talk 3 jokes about software developer"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))


# import torch

# print("Torch version:",torch.__version__)

# print("Is CUDA enabled?",torch.cuda.is_available())