

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['past', 'present', 'future']

# sub_tense

gpt_open_yes = [.66, .58, .53]
gpt_open_no= [.77, .67, .67]

gpt_5_yes = [.75, .646, .645]
gpt_5_no = [.686, .76, .642]

llama_70_yes = [.7, .651, .499]
llama_70_no = [.747, .706, .660]

llama_8_yes = [.463, 0.469, 0.376]
llama_8_no = [0.436, 0.517, 0.436]

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


plt.savefig('/home/lalady6977/Downloads/sub_tense.png', bbox_inches='tight')
