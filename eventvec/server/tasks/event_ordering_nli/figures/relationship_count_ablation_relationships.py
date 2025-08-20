

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
logical_only_before_sorted = [.9418, .943, .9464, .947]


X_axis = np.arange(len(X)) 
  
markersize=12
matplotlib.rcParams.update({'font.size': 12})

alpha = 0.7
plt.plot(X_axis, only_after_sim, color='green', marker='H', markersize=markersize, label = 'after_sim', linestyle='--', alpha=alpha)
plt.plot(X_axis, only_before_sim, color='green', marker='*', markersize=markersize,  label = 'before_sim', linestyle='--', alpha=alpha)
plt.plot(X_axis, only_before_after, color='green', marker='D', markersize=markersize,  label = 'before_after', linestyle='--', alpha=alpha)
plt.plot(X_axis, only_before, color='green', marker='v', markersize=markersize,  label = 'before', linestyle='--', alpha=alpha)
plt.plot(X_axis, only_before_sorted, color='purple', marker='s', markersize=markersize,  label = 'before_sorted', linestyle='--', alpha=alpha)
plt.plot(X_axis, logical_only_before_sorted, color='green', marker='^', markersize=markersize,  label = 'before_sorted (logical)', linestyle='-.', alpha=alpha)
plt.plot(X_axis, sort_relationships, color='purple', marker='P', markersize=markersize,  label = 'sorted all rels', linestyle='--', alpha=alpha)
plt.plot(X_axis, all_diff,  color='blue', marker='X', markersize=markersize, label = 'strict', linestyle='-', alpha=1)

ax = plt.gca()
ax.set_ylim([0.58, 1.0])
#ax.set_xlim([0.2, 0.5])

plt.grid()
plt.xticks(X_axis, X, rotation=45) 
plt.xlabel("Number of relationships in the\npremise") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', ncols=2, bbox_to_anchor=(0.5,-0.25)) 


plt.savefig('/home/lalady6977/Downloads/relationship_ablation_relationships.png', bbox_inches='tight')
