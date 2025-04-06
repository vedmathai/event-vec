

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['DCT-sub', 'DCT-matrix', 'matrix-sub']

# matrix_tense

gpt_past = [.551, .824, .628]
gpt_present= [.624, .821, .644]
gpt_future = [.56, .753, .62]

llama_past = [.564, .696, .667]
llama_present= [.651, .749, .703]
llama_future = [.599, .708, .689]

roberta_past = [.813, .996, .784]
roberta_present= [.799, .943, .807]
roberta_future = [.652, .556, .716]

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
plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.15), ncols=3) 


plt.savefig('/home/lalady6977/Downloads/matrix_tense.png', bbox_inches='tight')
