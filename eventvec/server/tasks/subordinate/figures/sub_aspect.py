

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib

font = {'size'   : 14}
matplotlib.rc('font', **font)
  
X = ['perfect', 'simple', 'perfect-continuous','continuous']

# matrix_tense

simple = 27.4 + 14.3 + 9.5+ 3.9 +3.9 + 1.9 + 0.3 + 0.8 + 0.4 + 0.4
perf = 1.5 + 12.6 + .4
continuous = 1 + 11.4 + 1.4 + 2.6 + .2

frequency = [perf/100, simple/100, continuous/100, continuous/100]

gpt_oss_120b = [.731, .667, .635, .604]
gpt_5 = [.738, .740, .654, .633 ]
llama_70 = [.754, .615, .721, .565]
llama_8 = [.592, .479, .340, .387]
deepseek = [.684, .654, .636, .585]
qwen3 = [.629, .468, .560, .511]


X_axis = np.arange(len(X)) 

fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
middle = 0.0
width = 0.05
#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

ax.plot(X_axis, frequency, color='C0', linestyle='dotted', label = 'gpt_oss_120b',  linewidth='3')
ax.plot(X_axis, gpt_oss_120b, color='C0', linestyle='solid', label = 'gpt_oss_120b',  linewidth='3')
ax.plot(X_axis, gpt_5, color='C1', linestyle='solid', label = 'GPT-5',  linewidth='3')
ax.plot(X_axis, llama_70, color='C2', linestyle='solid', label = 'Llama 70b',  linewidth='3')
ax.plot(X_axis, llama_8, color='C3', linestyle='solid', label = 'llama 8b',  linewidth='3')
ax.plot(X_axis, deepseek, color='C4', linestyle='solid', label = 'DeepSeek-R1',  linewidth='3')
ax.plot(X_axis, qwen3, color='C5', linestyle='solid', label = 'Qwen3',  linewidth='3')


ax = plt.gca()
ax.set_ylim([0, 0.85])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=10) 
plt.xlabel("Relationship") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.25), ncol=3) 


plt.savefig('/home/lalady6977/Downloads/sub_aspect.png', bbox_inches='tight')


