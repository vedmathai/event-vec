

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
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
only_before_sorted = [.964, .927, .872, .869, .661, .666]
logical = [.8, .75, .75, .77, .77, .67]
spatial = [.75, .62, .57, .56, .57, .44]
gpt = [.456, .433, .415, .443, .43, .409]
llama_405B = [.398, .406, .37, .384, .467, .383]


X_axis = np.arange(len(X)) 
markersize=12
matplotlib.rcParams.update({'font.size': 14})

#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')
plt.plot(X_axis, all_diff,  color='blue', marker='X', markersize=markersize, label = 'roberta_strict', linestyle='-')
#plt.plot(X_axis, same_english, color='C1', marker='o', markersize=markersize,  label = 'roberta_same_templates', linestyle='--')
#plt.plot(X_axis, same_names, color='C2', marker='v', markersize=markersize,  label = 'roberta_same_names', linestyle='--')
#plt.plot(X_axis, same_structures, color='C3', marker='^', markersize=markersize,  label = 'roberta_same_timelines', linestyle='--')

#plt.plot(X_axis, sort_relationships, color='C4', marker='<', markersize=markersize,  label = 'roberta_sort_relationship', linestyle='--')
#plt.plot(X_axis, only_after_sim, color='C5', marker='>', markersize=markersize, label = 'roberta_only_after_sim', linestyle='--')
#plt.plot(X_axis, only_before_sim, color='C6', marker='1', markersize=markersize,  label = 'roberta_only_before_sim', linestyle='--')
#plt.plot(X_axis, only_before_after, color='C7', marker='2', markersize=markersize,  label = 'roberta_before_after', linestyle='--')
#plt.plot(X_axis, only_before_sorted, color='C8', marker='3', markersize=markersize,  label = 'roberta_only_before_sorted', linestyle='--')
plt.plot(X_axis, logical, color='C9', marker='P', markersize=markersize,  label = 'RoBERTa logical', linestyle='--')
plt.plot(X_axis, spatial, color='C10', marker='H', markersize=markersize,  label = 'RoBERTa spatial', linestyle='-.')

plt.plot(X_axis, gpt, color='C11', marker='.',  label = 'gpt-4o (temporal)', linestyle='-', markersize=markersize)
plt.plot(X_axis, llama_405B, color='C12', marker='p', markersize=markersize,  label = 'llama-405B (temporal)', linestyle='-')


ax = plt.gca()
ax.set_ylim([0.3, .95])
#ax.set_xlim([0.2, 0.5])

plt.grid()
plt.xticks(X_axis, X, rotation=45) 
plt.xlabel("Number of hops in the premise") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', ncols=2, bbox_to_anchor=(0.5,-0.25)) 


plt.savefig('/home/lalady6977/Downloads/hops_ablation.png', bbox_inches='tight')
