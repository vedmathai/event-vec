
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['gpt-oss-120b', 'GPT-5', 'DeepSeek-R1']

# matrix_tense

font = {'size'   : 14}
matplotlib.rc('font', **font)

ate = [.659, .691, .640]
jogged = [.639, .735, .597]
fasted = [.616, .633, .599]
graduated = [.366, .436, .491]
retired = [.366, .429, .419]
died = [.463, 0.537,.465]


X_axis = np.arange(len(X)) 

fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
middle = 0.0
width = 0.05
#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

ax.bar(X_axis - (2.5 * width) - middle, ate, width=width, color='C0', align='center', hatch='\\\\', label = 'eat', alpha=1, edgecolor = "black")
ax.bar(X_axis - (1.5 * width) - middle, jogged, width=width, color='C0', align='center', hatch='\\', label = 'jog', alpha=1, edgecolor = "black")
ax.bar(X_axis - (0.5 * width) - middle, fasted, width=width, color='C0', align='center', hatch='//', label = 'fast', alpha=1, edgecolor = "black")
ax.bar(X_axis + (.5 * width) - middle, graduated, width=width, color='C1', align='center', hatch='\\\\', label = 'graduate', alpha=1, edgecolor = "black")
ax.bar(X_axis + (1.5 * width) - middle, retired, width=width, color='C1', align='center', hatch='\\', label = 'retire', alpha=1, edgecolor = "black")
ax.bar(X_axis + (2.5 * width) - middle, died, width=width, color='C1', align='center', hatch='//', label = 'die', alpha=1, edgecolor = "black")


ax = plt.gca()
ax.set_ylim([0.2, 0.85])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=0) 
plt.xlabel("Models") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.15), ncol=3) 


plt.savefig('/home/lalady6977/Downloads/verbs_comparison.png', bbox_inches='tight')

