import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl

font = {'size'   : 11}
matplotlib.rc('font', **font)

true2model = {
    'gpt-oss-120b':
        {
            "gpt-oss-120b": 1,
            "GPT-5": 0.648,
            "Llama-70B": .69,
            "Llama-8B": .535,
            "DeepSeek-R1": .668,
            'Qwen3': .627,
        },
    'GPT-5':
        {
            "gpt-oss-120b": .648,
            "GPT-5": 1,
            "Llama-70B": .682,
            "Llama-8B": .451,
            "DeepSeek-R1": .686,
            'Qwen3': .647,
        },
    'Llama-70B':
        {
            "gpt-oss-120b": .690,
            "GPT-5": .682,
            "Llama-70B": 1,
            "Llama-8B": .499,
            "DeepSeek-R1": .613,
            'Qwen3': .632,
        },
    'Llama-8B':
        {
            "gpt-oss-120b": .535,
            "GPT-5": .451,
            "Llama-70B": .499,
            "Llama-8B": 1,
            "DeepSeek-R1": .426,
            'Qwen3': .433,
        },
    'DeepSeek-R1':
        {
            "gpt-oss-120b": .668,
            "GPT-5": .686,
            "Llama-70B": .692,
            "Llama-8B": .426,
            "DeepSeek-R1": 1,
            'Qwen3': .613,
        },
    'Qwen3':
        {
            "gpt-oss-120b": .627,
            "GPT-5": .647,
            "Llama-70B": .632,
            "Llama-8B": .433,
            "DeepSeek-R1": .613,
            'Qwen3': 1,
        },
}


true_labels = ["gpt-oss-120b", "GPT-5", "Llama-70B", "Llama-8B", "DeepSeek-R1", 'Qwen3']
model_labels = ["Qwen3", "DeepSeek-R1", "Llama-8B", "Llama-70B", "GPT-5", "gpt-oss-120b"]

harvest = np.array([[true2model[a][b] for b in true_labels] for a in model_labels])


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# Show all ticks and label them with the respective list entries
ax.set_xticks(range(len(true_labels)), labels=true_labels,
              rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(model_labels)), labels=model_labels)

# Loop over data dimensions and create text annotations.
for i in range(len(true_labels)):
    for j in range(len(model_labels)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

plt.xlabel("True Labels") 
plt.ylabel("Predicted Labels") 

fig.tight_layout()
plt.savefig('/home/lalady6977/Downloads/model_model_agreement.png', bbox_inches='tight')
