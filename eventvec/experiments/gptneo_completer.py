from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

prompt = (
    """

"""
)

while True:
    prompt = input()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=20,
        num_return_sequences=10,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print(gen_text)
