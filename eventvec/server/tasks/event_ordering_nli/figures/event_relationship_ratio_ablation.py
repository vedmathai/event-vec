

import numpy as np  
import matplotlib.pyplot as plt  
  
X = ['<1', '==1', '>1' ]

# 5th percentile bucket

all_diff = [0.67, .75, .85]
same_english = [.71, .75, .86]
same_names = [.64, .73, .83]
same_structures = [.64, .732, .82]

sort_relationships = [.739, .812, .901]
only_after_sim = [.643, .704, .901]
only_before_sim = [.636, .695, .786]
only_before_after = [.795, .837, .968]
mpnet = [.492, .542, .562]

X_axis = np.arange(len(X)) 
  
markersize=7

#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')
plt.plot(X_axis, all_diff,  color='C0', marker='.',  label = 'roberta_all_different', linestyle='--')
plt.plot(X_axis, same_english, color='C1', marker='o', label = 'roberta_same_templates', linestyle='--')
plt.plot(X_axis, same_names, color='C2', marker='v', label = 'roberta_same_names', linestyle='--')
plt.plot(X_axis, same_structures, color='C3', marker='^', label = 'roberta_same_timelines', linestyle='--')

plt.plot(X_axis, sort_relationships, color='C4', marker='<', label = 'roberta_sort_relationship', linestyle='--')
plt.plot(X_axis, only_after_sim, color='C5', marker='>', markersize=5, label = 'roberta_only_after_sim', linestyle='--')
plt.plot(X_axis, only_before_sim, color='C6', marker='1', label = 'roberta_only_before_sim', linestyle='--')
plt.plot(X_axis, only_before_after, color='C7', marker='2', label = 'roberta_before_after', linestyle='--')
plt.plot(X_axis, mpnet, color='C8', marker='s', label = 'mpnet', linestyle='--')

ax = plt.gca()
ax.set_ylim([0.45, 1.0])
#ax.set_xlim([0.2, 0.5])

plt.grid()
plt.xticks(X_axis, X, rotation=45) 
plt.xlabel("Event to relationship ratio in the premise") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.15)) 

plt.savefig('/home/lalady6977/Downloads/event-rel-ration.png', bbox_inches='tight')
