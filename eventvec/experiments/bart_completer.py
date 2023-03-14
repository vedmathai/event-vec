from transformers import BartForConditionalGeneration, BartTokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
tok = BartTokenizer.from_pretrained("facebook/bart-large")



def complete(sentence):
    example_english_phrase = sentence
    batch = tok(example_english_phrase, return_tensors="pt", add_special_tokens=False)
    generated_ids = model.generate(batch["input_ids"], max_length=64, do_sample=True, num_beams=1, top_p=0.1, repetition_penalty=10.0)
    completed = tok.batch_decode(generated_ids, skip_special_tokens=False)
    return completed

sentences = [
    "Jack studied in college, to do this he"
]

for s in sentences:
    print(complete(s))