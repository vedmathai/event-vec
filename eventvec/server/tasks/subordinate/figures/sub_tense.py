

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['DCT-sub', 'DCT-matrix', 'matrix-sub']


# sub_tense

gpt_past = [.63, .712, .722]
gpt_present= [.612, .719, .625]
gpt_future = [.56, .753, .62]

llama_past = [.63, .674, .786]
llama_present= [.617, .689, .714]
llama_future = [.599, .708, .689]

roberta_past = [.723, .818, .786]
roberta_present= [.703, .833, .747]
roberta_future = [.807, .856, .735]

X_axis = np.arange(len(X)) 

fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
middle = 0.0
width = 0.05
#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

ax.bar(X_axis - (5 * width) + middle, gpt_past, width=width, color='C0', align='center', hatch='//', label = 'gpt_past')
ax.bar(X_axis - (4 * width) + middle, gpt_present, width=width, color='C1', align='center', hatch='//', label = 'gpt_present')
ax.bar(X_axis - (3 * width) + middle, gpt_future, width=width, color='C2', align='center', hatch='//', label = 'gpt_future')

ax.bar(X_axis - (1 * width), llama_past, width=width, color='C0', align='center', label = 'llama_past')
ax.bar(X_axis - (0 * width), llama_present, width=width, color='C1', align='center', label = 'llama_present')
ax.bar(X_axis + (1 * width), llama_future, width=width, color='C2', align='center', label = 'llama_future')

ax.bar(X_axis + (3 * width) - middle, roberta_past, width=width, color='C0', align='center', hatch='\\', label = 'roberta_past')
ax.bar(X_axis + (4 * width) - middle, roberta_present, width=width, color='C1', align='center', hatch='\\', label = 'roberta_present')
ax.bar(X_axis + (5 * width) - middle, roberta_future, width=width, color='C2', align='center', hatch='\\', label = 'roberta_future')


ax = plt.gca()
ax.set_ylim([0.5, 1])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=0) 
plt.xlabel("Relationship") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.15), ncols=2) 


plt.savefig('/home/lalady6977/Downloads/sub_tense.png', bbox_inches='tight')
