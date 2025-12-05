

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['gpt-oss-\n120b', 'GPT 5', 'DeepSeek-R1', 'Qwen3', 'Llama_70', 'Llama_8']

font = {'size'   : 14}
matplotlib.rc('font', **font)
# matrix_tense

unembedded = [.7, .740, .725, .733, .657, .461]
cloze = [.647, .745, .667, .672, .667, .4]
qa = [.659, .691, .640, .546, .664, .449]

X_axis = np.arange(len(X)) 

fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
middle = 0.0
width = 0.2
#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

ax.bar(X_axis - (1 * width) - middle, unembedded, width=width, color='black', align='center', hatch='\\', label = 'Unembedded', alpha=1)
ax.bar(X_axis - (0 * width) - middle, cloze, width=width, color='C0', align='center', hatch='\\', label = 'Cloze', alpha=1)
ax.bar(X_axis + (1 * width) - middle, qa, width=width, color='C1', align='center', hatch='\\', label = 'Dialogue-question', alpha=1)

ax = plt.gca()
ax.set_ylim([0.3, 0.85])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=18) 
plt.xlabel("Models") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.3), ncol=3) 


plt.savefig('/home/lalady6977/Downloads/models_prompts_comparison.png', bbox_inches='tight')
