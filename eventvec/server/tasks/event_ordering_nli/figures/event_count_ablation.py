

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
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
only_before_sorted = [.95, .944, .94, .96]
logical = [.78, .77, .79, .81]
spatial = [.66, .65, .66, .66]
gpt = [.527, .483, .389, .395]
llama_405B = [.407, .409, .395, .372]


X_axis = np.arange(len(X)) 
markersize=12
matplotlib.rcParams.update({'font.size': 14})
  
plt.plot(X_axis, all_diff,  color='blue', marker='X', markersize=markersize, label = 'RoBERTa strict', linestyle='-')
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
ax.set_ylim([0.3, 1])
#ax.set_xlim([0.2, 0.5])

plt.grid()
plt.xticks(X_axis, X, rotation=45)
plt.xlabel("Number of events in the\npremise")
plt.ylabel("Model Macro-F1 scores")
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', ncols=2, bbox_to_anchor=(0.5,-0.25)) 
plt.savefig('/home/lalady6977/Downloads/events_ablation.png', bbox_inches='tight')
