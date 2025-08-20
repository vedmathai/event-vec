

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['4', '8', '16', '32' ]

# 5th percentile bucket

all_diff = [0.838, .80, .77, .64]
same_english = [.85, .813, .8, .70]
same_names = [.806, .76, .75, .63]
same_structures = [.821, .752, .752, .62]


X_axis = np.arange(len(X)) 
  
markersize=12
matplotlib.rcParams.update({'font.size': 14})
alpha=0.7

#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')
plt.plot(X_axis, same_english, color='blue', marker='X', markersize=markersize,  label = 'same_templates', linestyle='-', alpha=alpha)
plt.plot(X_axis, same_names, color='green', marker='P', markersize=markersize,  label = 'same_names', linestyle='--', alpha=alpha)
plt.plot(X_axis, same_structures, color='purple', marker='H', markersize=markersize,  label = 'same_timelines', linestyle='--', alpha=alpha)
plt.plot(X_axis, all_diff,  color='red', marker='o', markersize=markersize, label = 'RoBERTa_strict', linestyle='--', alpha=alpha)

ax = plt.gca()
ax.set_ylim([0.58, .9])
#ax.set_xlim([0.2, 0.5])

plt.grid()
plt.xticks(X_axis, X, rotation=45) 
plt.xlabel("Number of relationships in the\npremise") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', ncols=2, bbox_to_anchor=(0.5,-0.25)) 



plt.savefig('/home/lalady6977/Downloads/relationship_ablation_same_one.png', bbox_inches='tight')
