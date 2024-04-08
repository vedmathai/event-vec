# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

access_token = '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'

tokenizer = AutoTokenizer.from_pretrained("transformers_cache/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("transformers_cache/Llama-2-7b-hf")

tokenizer.save_pretrained("transformers_cache/meta-llama/Llama-2-7b-hf")
model.save_pretrained("transformers_cache/Llama-2-7b-hf")