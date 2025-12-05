
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib

font = {'size'   : 16}
matplotlib.rc('font', **font)
  
X = ['past', 'present', 'future']

# matrix_tense

gpt_open_simple= [.8, .750, 1]



gpt_open_dct_matrix_direct = [1, .288, .397]
gpt_open_dct_matrix_indirect= [1, .362, .423]

gpt_open_dct_sub_direct = [0.697, .581, 0.465]
gpt_open_dct_sub_direct_replaced = [0.768, .685, 0.649]
gpt_open_dct_sub_indirect= [0.767, 0.642, 0.642]

gpt_open_matrix_sub_direct = [0.545, .738, .452]
gpt_open_matrix_sub_indirect = [0.688, .766, .684]

X_axis = np.arange(len(X)) 

fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
middle = 0.0
width = 0.06
#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

ax.bar(X_axis - (4.5 * width) - middle, gpt_open_simple, width=width, color='black', align='center', hatch='*', label = 'unembedded', alpha=1, edgecolor = "black")

ax.bar(X_axis - (3.1 * width) - middle, gpt_open_dct_matrix_direct, width=width, color='C0', align='center', hatch='\\\\', label = 'dct_matrix_direct', alpha=0.4, edgecolor = "black")
ax.bar(X_axis - (2 * width) - middle, gpt_open_dct_matrix_indirect, width=width, color='C0', align='center', hatch='//', label = 'dct_matrix_indirect', alpha=0.4, edgecolor = "black")

ax.bar(X_axis - (0.6 * width) - middle, gpt_open_dct_sub_direct, width=width, color='C1', align='center', hatch='\\\\', label = 'dct_sub_direct', alpha=1, edgecolor = "black")
ax.bar(X_axis - (0.6 * width) - middle, gpt_open_dct_sub_direct_replaced, width=width, color='C1', align='center', hatch='\\\\', label = 'dct_sub_direct_replaced', alpha=.3, edgecolor = "black")
ax.bar(X_axis + (0.6 * width) - middle, gpt_open_dct_sub_indirect, width=width, color='C1', align='center', hatch='//', label = 'dct_sub_indirect', alpha=1, edgecolor = "black")

ax.bar(X_axis + (2 * width) - middle, gpt_open_matrix_sub_direct, width=width, color='C2', align='center', hatch='\\\\', label = 'matrix_sub_direct', alpha=0.4, edgecolor = "black")
ax.bar(X_axis + (3.1 * width) - middle, gpt_open_matrix_sub_indirect, width=width, color='C2', align='center', hatch='//', label = 'matrix_sub_indirect', alpha=0.4,edgecolor = "black")

ax = plt.gca()
ax.set_ylim([0.2, 1.05])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=0) 
plt.xlabel("Relationships") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.20), ncol = 2) 


plt.savefig('/home/lalady6977/Downloads/relationship_comparison.png', bbox_inches='tight')
