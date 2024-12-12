

import numpy as np  
import matplotlib.pyplot as plt  
  
X = ['1', '2', '3', '4', '5', '6']

# 5th percentile bucket

all_diff = [0.837, .736, .662, .662, .71, .635]
same_english = [.85, .77, .68, .74, .66, .621]
same_names = [.806, .691, .656, .681, .698, .645]
same_structures = [.803, .69, .65, .675, .667, .672]

sort_relationships = [.868, .809, .792, .786, .690, .691]
only_after_sim = [.749, .713, .666, .613, 0.638, .6744]
only_before_sim = [.769, .720, .666, .666, .648, .599]
only_before_after = [.904, .80, .75, .69, .60, float('NaN')]
mpnet = [.609, .536, .483, .49, .451, .438]


X_axis = np.arange(len(X)) 

markersize=7

#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')
plt.plot(X_axis, all_diff,  color='C0', marker='.', markersize=markersize, label = 'roberta_all_different', linestyle='--')
plt.plot(X_axis, same_english, color='C1', marker='o', markersize=markersize,  label = 'roberta_same_templates', linestyle='--')
plt.plot(X_axis, same_names, color='C2', marker='v', markersize=markersize,  label = 'roberta_same_names', linestyle='--')
plt.plot(X_axis, same_structures, color='C3', marker='^', markersize=markersize,  label = 'roberta_same_timelines', linestyle='--')

plt.plot(X_axis, sort_relationships, color='C4', marker='<', markersize=markersize,  label = 'roberta_sort_relationship', linestyle='--')
plt.plot(X_axis, only_after_sim, color='C5', marker='>', markersize=markersize, label = 'roberta_only_after_sim', linestyle='--')
plt.plot(X_axis, only_before_sim, color='C6', marker='1', markersize=markersize,  label = 'roberta_only_before_sim', linestyle='--')
plt.plot(X_axis, only_before_after, color='C7', marker='2', markersize=markersize,  label = 'roberta_before_after', linestyle='--')
plt.plot(X_axis, mpnet, color='C8', marker='s', markersize=markersize,  label = 'mpnet', linestyle='--')


ax = plt.gca()
ax.set_ylim([0.4, .95])
#ax.set_xlim([0.2, 0.5])

plt.grid()
plt.xticks(X_axis, X, rotation=45) 
plt.xlabel("Number of hops in the premise") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.15)) 

plt.savefig('/home/lalady6977/Downloads/hops_ablation.png', bbox_inches='tight')
