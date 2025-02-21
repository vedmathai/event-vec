

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['<1', '==1', '>1' ]

# 5th percentile bucket

all_diff = [.85, .75, 0.67]
same_english = [.86, .75, .71]
same_names = [.83, .73, .64]
same_structures = [.82, .732, .64]

sort_relationships = [.901, .812, .739]
only_after_sim = [.901, .704, .643]
only_before_sim = [.786, .695, .636]
only_before_after = [.968, .837, .795]
llama_405B = [.426, .386, .368]
gpt = [.511, .447, .402]

X_axis = np.arange(len(X)) 
  
markersize=12
matplotlib.rcParams.update({'font.size': 20})

#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')
plt.plot(X_axis, all_diff,  color='C0', marker='.',  label = 'roberta_standard', linestyle='--', markersize=markersize)
plt.plot(X_axis, same_english, color='C1', marker='o', label = 'roberta_same_templates', linestyle='--', markersize=markersize)
plt.plot(X_axis, same_names, color='C2', marker='v', label = 'roberta_same_names', linestyle='--', markersize=markersize)
plt.plot(X_axis, same_structures, color='C3', marker='^', label = 'roberta_same_timelines', linestyle='--', markersize=markersize)

plt.plot(X_axis, sort_relationships, color='C4', marker='<', label = 'roberta_sort_relationship', linestyle='--', markersize=markersize)
plt.plot(X_axis, only_after_sim, color='C5', marker='>', label = 'roberta_only_after_sim', linestyle='--', markersize=markersize)
plt.plot(X_axis, only_before_sim, color='C6', marker='1', label = 'roberta_only_before_sim', linestyle='--', markersize=markersize)
plt.plot(X_axis, only_before_after, color='C7', marker='2', label = 'roberta_before_after', linestyle='--', markersize=markersize)
plt.plot(X_axis, gpt, color='C8', marker='3',  label = 'gpt', linestyle='--', markersize=markersize)
plt.plot(X_axis, llama_405B, color='C9', marker='4', markersize=markersize,  label = 'llama-405B', linestyle='--')

ax = plt.gca()
ax.set_ylim([0.35, 1.0])
#ax.set_xlim([0.2, 0.5])


plt.grid()
plt.xticks(X_axis, X, rotation=45) 
plt.xlabel("Relationships to events ratio\nin the premise") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
#plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.15)) 

plt.savefig('/home/lalady6977/Downloads/event_rel_ablation.png', bbox_inches='tight')
