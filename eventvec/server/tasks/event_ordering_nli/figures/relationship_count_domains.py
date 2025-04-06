

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['4', '8', '16', '32']

# 5th percentile bucket

all_diff = [0.838, .80, .77, .64]
logical = [.831, .79, .785, .78]
spatial = [.6322, .65, .65, .632]


X_axis = np.arange(len(X))

markersize=12
matplotlib.rcParams.update({'font.size': 14})

plt.plot(X_axis, logical, color='blue', marker='P', markersize=markersize,  label = 'logical', linestyle='--')
plt.plot(X_axis, spatial, color='green', marker='H', markersize=markersize,  label = 'spatial', linestyle='--')
plt.plot(X_axis, all_diff,  color='red', marker='o', markersize=markersize, label = 'temporal', linestyle='--')

ax = plt.gca()
ax.set_ylim([0.4, 1.0])
#ax.set_xlim([0.2, 0.5])

plt.grid()
plt.xticks(X_axis, X, rotation=45) 
plt.xlabel("Number of relationships in the\npremise") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='lower left', ncols=2) 


plt.savefig('/home/lalady6977/Downloads/relationship_ablation_domains.png', bbox_inches='tight')
