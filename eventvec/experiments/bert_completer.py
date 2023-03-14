# instantiate sentence fusion model
from transformers import EncoderDecoderModel, AutoTokenizer
sentence_fuser = EncoderDecoderModel.from_pretrained("google/roberta2roberta_L-24_discofuse")
tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_discofuse")

input_ids = tokenizer(
    "This is the first sentence. Therefore this is the...", add_special_tokens=False, return_tensors="pt"
).input_ids

outputs = sentence_fuser.generate(input_ids, max_length=64, do_sample=True, num_beams=3, top_p=0.5, repetition_penalty=10.0)

print(tokenizer.decode(outputs[0]))