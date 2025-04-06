

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['4', '8', '16', '32' ]

# 5th percentile bucket

all_diff = [0.838, .80, .77, .64]

sort_relationships = [.881, .825, .851, .766]
only_after_sim = [.811, .741, .731, .616]
only_before_sim = [.801, .740, .728, .605]
only_before_after = [.945, .91, .867, .751]
only_before_sorted = [.975, .9488, .944, .933]
only_before = [.95, .952, .953, .958]


X_axis = np.arange(len(X)) 
  
markersize=12
matplotlib.rcParams.update({'font.size': 12})


plt.plot(X_axis, sort_relationships, color='C1', marker='P', markersize=markersize,  label = 'sort_relationship', linestyle='--')
plt.plot(X_axis, only_after_sim, color='C2', marker='H', markersize=markersize, label = 'only_after_sim', linestyle='--')
plt.plot(X_axis, only_before_sim, color='C3', marker='*', markersize=markersize,  label = 'only_before_sim', linestyle='--')
plt.plot(X_axis, only_before_after, color='C4', marker='D', markersize=markersize,  label = 'before_after', linestyle='--')
plt.plot(X_axis, only_before_sorted, color='C5', marker='s', markersize=markersize,  label = 'only_before_sorted', linestyle='--')
plt.plot(X_axis, only_before, color='C6', marker='v', markersize=markersize,  label = 'only_before', linestyle='--')
plt.plot(X_axis, all_diff,  color='red', marker='o', markersize=markersize, label = 'strict', linestyle='--')

ax = plt.gca()
ax.set_ylim([0.3, 1.0])
#ax.set_xlim([0.2, 0.5])

plt.grid()
plt.xticks(X_axis, X, rotation=45) 
plt.xlabel("Number of relationships in the\npremise") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='lower left', ncols=2) 

plt.savefig('/home/lalady6977/Downloads/relationship_ablation_relationships.png', bbox_inches='tight')
