import torch
from transformers import BigBirdForQuestionAnswering, BigBirdTokenizer

# by default its in `block_sparse` mode with num_random_blocks=3, block_size=64
#model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-base-trivia-itc")

# you can change `attention_type` to full attention like this:
# model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-base-trivia-itc", attention_type="original_full")

# you can change `block_size` & `num_random_blocks` like this:
model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-base-trivia-itc", block_size=16, num_random_blocks=2)
tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-base-trivia-itc")



question = '''Who was the spouse of Agnes of Courtenay from 1148 to Jun 1149?'''

paragraph = ''' '''
            
encoding = tokenizer.encode_plus(text=question,text_pair=paragraph)

encoded_input = tokenizer(question, paragraph, return_tensors='pt')
output = model(**encoded_input)

inputs = encoding['input_ids']  #Token embeddings
tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens

start_index = torch.argmax(output.start_logits)
start_value = torch.max(output.start_logits)

end_index = torch.argmax(output.end_logits)
end_value = torch.max(output.end_logits)
print(start_index, end_index)

answer = ' '.join(tokens[start_index:end_index+1])

corrected_answer = ''

for word in answer.split():
    
    #If it's a subword token
    if word[0:2] == '▁':
        corrected_answer += word[1:]
    else:
        corrected_answer += ' ' + word

print(corrected_answer)
print(start_value, end_value)
