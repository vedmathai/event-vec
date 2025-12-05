

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
font = {'size'   : 14}
matplotlib.rc('font', **font)

X = ['past', 'present', 'future']

# matrix_tense


gpt_oss_120b = [.633, .743, .601]
gpt_5 = [.730, .787, .544]
llama_70 = [.702, .750, .539]
llama_8 = [.479, .489, .380]
deepseek = [.629, .756, .533]
qwen3 = [.551, .684, .404]


X_axis = np.arange(len(X)) 

fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
middle = 0.0
width = 0.05
#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

ax.plot(X_axis, gpt_oss_120b, color='C0', linestyle='solid', label = 'gpt-oss-120b',  linewidth='2')
ax.plot(X_axis, gpt_5, color='C1', linestyle='solid', label = 'GPT-5',  linewidth='2')
ax.plot(X_axis, llama_70, color='C2', linestyle='solid', label = 'Llama 70b',  linewidth='2')
ax.plot(X_axis, llama_8, color='C3', linestyle='solid', label = 'llama 8b',  linewidth='2')
ax.plot(X_axis, deepseek, color='C4', linestyle='solid', label = 'DeepSeek-R1',  linewidth='2')
ax.plot(X_axis, qwen3, color='C5', linestyle='solid', label = 'Qwen3',  linewidth='2')



ax = plt.gca()
ax.set_ylim([0.3, 0.85])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=0) 
plt.xlabel("Relationship") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.15), ncol=3) 


plt.savefig('/home/lalady6977/Downloads/matrix_tense.png', bbox_inches='tight')
