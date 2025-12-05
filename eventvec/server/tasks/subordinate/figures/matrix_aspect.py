

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['perfect', 'simple', 'continuous', 'perfect-continuous']

# matrix_tense

gpt_open_yes = [0.589, .628, 0.566, 0.553]
gpt_open_no= [0.653, 0.702, 0.701, 0.683]

gpt_5_yes = [.691, .703, .670, .587]
gpt_5_no = [.662, .656, .718, .656]

llama_70_yes = [.634, .610, .664, .613]
llama_70_no = [.722, .664, .679, .749]

llama_8_yes = [.447, .508, .445, .366]
llama_8_no = [.470, .493, .456, .417]

X_axis = np.arange(len(X)) 

fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
middle = 0.0
width = 0.05
#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

ax.plot(X_axis, gpt_open_yes, color='C0', linestyle='dashed', label = 'gpt_open_yes', alpha=1)
ax.plot(X_axis, gpt_open_no, color='C0', linestyle='solid', label = 'gpt_open_no', alpha=0.7)

ax.plot(X_axis, gpt_5_yes, color='C1', linestyle='dashed', label = 'gpt_5_yes', alpha=1)
ax.plot(X_axis, gpt_5_no, color='C1', linestyle='solid', label = 'gpt_5_no', alpha=0.7)

ax.plot(X_axis, llama_70_yes, color='C2', linestyle='dashed', label = 'llama_70_yes', alpha=1)
ax.plot(X_axis, llama_70_no, color='C2', linestyle='solid', label = 'llama_70_no', alpha=0.7)

ax.plot(X_axis, llama_8_yes, color='C3', linestyle='dashed', label = 'llama_8_yes', alpha=1)
ax.plot(X_axis, llama_8_no, color='C3', linestyle='solid', label = 'llama_8_no', alpha=0.7)

ax = plt.gca()
ax.set_ylim([0.4, 0.85])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=0) 
plt.xlabel("Relationship") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.15)) 


plt.savefig('/home/lalady6977/Downloads/matrix_aspect.png', bbox_inches='tight')
