from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch


class GPTQuestionGenerator:
    def __init__(self):
        self._model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
        self._tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    def generate(self, prompt):
        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids

        gen_tokens = self._model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=100,
        )
        gen_text = self._tokenizer.batch_decode(gen_tokens)[0]
        return gen_text
