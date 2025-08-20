

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['4', '8', '16', '32']

# 5th percentile bucket

all_diff = [0.838, .80, .77, .64]
logical = [.831, .79, .785, .78]
spatial = [.6322, .65, .65, .632]
gpt = [.569, .447, .418, .365]
llama_405B = [.417, .42, .405, 0.335]


X_axis = np.arange(len(X))

markersize=12
matplotlib.rcParams.update({'font.size': 14})

plt.plot(X_axis, all_diff,  color='blue', marker='X', markersize=markersize, label = 'roberta_strict', linestyle='-')
plt.plot(X_axis, logical, color='C9', marker='P', markersize=markersize,  label = 'RoBERTa logical', linestyle='--')
plt.plot(X_axis, spatial, color='C10', marker='H', markersize=markersize,  label = 'RoBERTa spatial', linestyle='-.')

plt.plot(X_axis, gpt, color='C11', marker='.',  label = 'gpt-4o (temporal)', linestyle='-', markersize=markersize)
plt.plot(X_axis, llama_405B, color='C12', marker='p', markersize=markersize,  label = 'llama-405B (temporal)', linestyle='-')

ax = plt.gca()
ax.set_ylim([0.2, 1.0])
#ax.set_xlim([0.2, 0.5])

plt.grid()
plt.xticks(X_axis, X, rotation=45) 
plt.xlabel("Number of relationships in the\npremise") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', ncols=2, bbox_to_anchor=(0.5,-0.25)) 



plt.savefig('/home/lalady6977/Downloads/relationship_ablation_domains.png', bbox_inches='tight')
