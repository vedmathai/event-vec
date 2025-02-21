

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['dct is _ sub', 'dct is _ matrix', 'matrix is _ sub']

# matrix_aspect

gpt_simple = [.621, .828, .626]
gpt_perfect= [.635, .722, .622]
gpt_continuous = [.591, .821, .631]
gpt_perfect_continuous = [.56, .753, .62]

llama_simple = [.584, .81, .696]
llama_perfect= [.6, .685, .676]
llama_continuous = [.587, .753, .705]
llama_perfect_continuous = [.599, .708, .689]

X_axis = np.arange(len(X)) 

fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
middle = 0.03
#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

ax.bar(X_axis-0.4 + middle, gpt_simple, width=0.1, color='C0', align='center', hatch='//', label = 'gpt_simple')
ax.bar(X_axis-0.3 + middle, gpt_perfect, width=0.1, color='C1', align='center', hatch='//', label = 'gpt_perfect')
ax.bar(X_axis-0.2 + middle, gpt_continuous, width=0.1, color='C2', align='center', hatch='//', label = 'gpt_continuous')
ax.bar(X_axis-0.1 + middle + 0., gpt_perfect_continuous, width=0.1, color='C3', align='center', hatch='//', label = 'gpt_perfect_continuous')

ax.bar(X_axis+0.1 - middle, llama_simple, width=0.1, color='C0', align='center', label = 'llama_simple')
ax.bar(X_axis+0.2 - middle, llama_perfect, width=0.1, color='C1', align='center', label = 'llama_perfect')
ax.bar(X_axis+0.3 - middle, llama_continuous, width=0.1, color='C2', align='center', label = 'llama_continuous')
ax.bar(X_axis+0.4 - middle, llama_perfect_continuous, width=0.1, color='C3', align='center', label = 'llama_perfect_continuous')


ax = plt.gca()
ax.set_ylim([0.45, 1])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=0) 
plt.xlabel("Matrix Aspect") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper left', ncol=1) 

plt.savefig('/home/lalady6977/Downloads/matrix_aspect.png', bbox_inches='tight')
