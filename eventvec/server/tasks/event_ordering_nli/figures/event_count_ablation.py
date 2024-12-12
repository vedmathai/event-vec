

import numpy as np  
import matplotlib.pyplot as plt  
  
X = ['4', '8', '16', '32' ]

# 5th percentile bucket

plain_llama = [0.187, .191, .187, 0.182]
all_diff = [0.786, .797, .755, .713]
same_english = [.814, .81, .77, .717]
same_names = [.737, .762, .75, .679]
same_structures = [.74, .76, .73, .68]
sort_relationships = [.835, .84, .802, .83]
only_after_sim = [.746, .732, .711, .686]
only_before_sim = [.731, .749, .712, .665]
only_before_after = [.901, .887, .854, .829]
mpnet = [.554, .52, .565, .546]


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
ax.set_ylim([0.5, 1])
#ax.set_xlim([0.2, 0.5])

plt.grid()
plt.xticks(X_axis, X, rotation=45) 
plt.xlabel("Number of events in the premise") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.15)) 
plt.savefig('/home/lalady6977/Downloads/events_ablation.png', bbox_inches='tight')
