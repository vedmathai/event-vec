

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['dct is _ sub', 'dct is _ matrix', 'matrix is _ sub']

# matrix_tense

gpt_past = [.551, .824, .628]
gpt_present= [.624, .821, .644]
gpt_future = [.56, .753, .62]

llama_past = [.564, .696, .667]
llama_present= [.651, .749, .703]
llama_future = [.599, .708, .689]

X_axis = np.arange(len(X)) 

fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
middle = 0.03
#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

ax.bar(X_axis-0.3 + middle, gpt_past, width=0.1, color='C0', align='center', hatch='//', label = 'gpt_past')
ax.bar(X_axis-0.2 + middle, gpt_present, width=0.1, color='C1', align='center', hatch='//', label = 'gpt_present')
ax.bar(X_axis-0.1 + middle, gpt_future, width=0.1, color='C2', align='center', hatch='//', label = 'gpt_future')

ax.bar(X_axis+0.1 - middle, llama_past, width=0.1, color='C0', align='center', label = 'llama_past')
ax.bar(X_axis+0.2 - middle, llama_present, width=0.1, color='C1', align='center', label = 'llama_present')
ax.bar(X_axis+0.3 - middle, llama_future, width=0.1, color='C2', align='center', label = 'llama_future')


ax = plt.gca()
ax.set_ylim([0.5, 1])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=0) 
plt.xlabel("Matrix Tense") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', ncol=3) 

plt.savefig('/home/lalady6977/Downloads/matrix_tense.png', bbox_inches='tight')
