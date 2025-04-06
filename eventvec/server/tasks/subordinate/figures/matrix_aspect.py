

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['DCT-sub', 'DCT-matrix', 'matrix-sub']

# matrix_aspect

gpt_simple = [.621, .828, .626]
gpt_perfect= [.635, .722, .622]
gpt_continuous = [.591, .821, .631]
gpt_perfect_continuous = [.56, .753, .62]

llama_simple = [.584, .81, .696]
llama_perfect= [.6, .685, .676]
llama_continuous = [.587, .753, .705]
llama_perfect_continuous = [.599, .708, .689]

roberta_simple = [.771, .969, .794]
roberta_perfect= [.72, .712, .751]
roberta_continuous = [.771, .969, .794]
roberta_perfect_continuous = [.745, .716, .733]


X_axis = np.arange(len(X)) 

fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
middle = 0.0
width = 0.05
#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

ax.bar(X_axis - (6.5 * width) + middle, gpt_simple, width=width, color='C0', align='center', hatch='//', label = 'gpt_simple')
ax.bar(X_axis - (5.5 * width) + middle, gpt_perfect, width=width, color='C1', align='center', hatch='//', label = 'gpt_perfect')
ax.bar(X_axis - (4.5 * width) + middle, gpt_continuous, width=width, color='C2', align='center', hatch='//', label = 'gpt_continuous')
ax.bar(X_axis - (3.5 * width) + middle, gpt_perfect_continuous, width=width, color='C3', align='center', hatch='//', label = 'gpt_perfect_continuous')

ax.bar(X_axis - (1.5 * width), llama_simple, width=width, color='C0', align='center', label = 'llama_simple')
ax.bar(X_axis - (0.5 * width), llama_perfect, width=width, color='C1', align='center', label = 'llama_perfect')
ax.bar(X_axis + (0.5 * width), llama_continuous, width=width, color='C2', align='center', label = 'llama_continuous')
ax.bar(X_axis + (1.5 * width), llama_perfect_continuous, width=width, color='C3', align='center', label = 'llama_perfect_continuous')

ax.bar(X_axis + (3.5 * width) - middle, roberta_simple, width=width, color='C0', align='center', hatch='\\', label = 'roberta_simple')
ax.bar(X_axis + (4.5 * width) - middle, roberta_perfect, width=width, color='C1', align='center', hatch='\\', label = 'roberta_perfect')
ax.bar(X_axis + (5.5 * width) - middle, roberta_continuous, width=width, color='C2', align='center', hatch='\\', label = 'roberta_continuous')
ax.bar(X_axis + (6.5 * width) - middle, roberta_perfect_continuous, width=width, color='C3', align='center', hatch='\\', label = 'roberta_perfect_continuous')

ax = plt.gca()
ax.set_ylim([0.45, 1])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=0) 
plt.xlabel("Relationship")
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.15), ncols=2) 


plt.savefig('/home/lalady6977/Downloads/matrix_aspect.png', bbox_inches='tight')
