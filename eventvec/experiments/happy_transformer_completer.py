from happytransformer import HappyGeneration
from happytransformer import GENSettings


gpt_neo = HappyGeneration(model_type="GPT-NEO", model_name="EleutherAI/gpt-neo-125M") 


top_k_sampling_settings = GENSettings(do_sample=True, top_k=50, max_length=30, min_length=10)

while True:
    prompt = input()
    output_top_k_sampling = gpt_neo.generate_text(prompt, args=top_k_sampling_settings)
    print (output_top_k_sampling.text)