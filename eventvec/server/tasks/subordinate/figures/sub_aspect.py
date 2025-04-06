

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['DCT-sub', 'DCT-matrix', 'matrix-sub']


# sub_aspect

gpt_simple = [.621, .743, .697]
gpt_perfect= [.635, .744, .687]
gpt_continuous = [.591, .747, .642]
gpt_perfect_continuous = [.56, .753, .62]

llama_simple = [.632, .688, .764]
llama_perfect= [.647, .706, .712]
llama_continuous = [.616, .698, .738]
llama_perfect_continuous = [.599, .708, .689]

roberta_simple = [.723, .803, .806]
roberta_perfect= [.7, .85, .715]
roberta_continuous = [.724, .826, .76]
roberta_perfect_continuous = [.838, .872, .74]

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
ax.set_ylim([0.43, 1])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=0) 
plt.xlabel("Relationship") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.15), ncols=2) 


plt.savefig('/home/lalady6977/Downloads/sub_aspect.png', bbox_inches='tight')
