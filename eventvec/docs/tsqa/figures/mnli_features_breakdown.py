

import numpy as np  
import matplotlib.pyplot as plt  
  
X = ['Modal', 'Is\nspeech', 'Is\nbelief', 'Sub of\nSpeech', 
     'Sub of\nBelief', 'Sub of\nIf', 'Negation', 'No\nFeature']

"""
# 4th and 5th percentile buckets
alone = [.276, .251, .255, .268, .27, 0.268, .309, .282]
rules = [.331, .314, .368, .321, .294, .313, .334, .313]
r_r = [.25, .267, .238, .258, .247, .305, .258, .261]

"""
# 5th percentile bucket
r_r = [0.23, 0.203, 0.248, 0.223, 0.246, 0.318, 0.249, 0.245]
rules = [0.335, 0.309, 0.379, 0.321, 0.328, 0.31, 0.359, 0.313]
alone = [.263, 0.228, 0.219, 0.248, 0.268, 0.281, 0.314, 0.287]

X_axis = np.arange(len(X)) 
  
multiplier = 2.8
plt.bar(X_axis - 0.1 * multiplier, r_r, 0.1 * multiplier, label = 'DistlBERT + RoBERTa')
plt.bar(X_axis - 0, alone, 0.1 * multiplier, label = 'RoBERTa alone')
plt.bar(X_axis + .1 * multiplier, rules, 0.1 * multiplier, label = 'Rules + RoBERTa')
ax = plt.gca()
ax.set_ylim([0.2, 0.5])
plt.xticks(X_axis, X) 
plt.xlabel("Linguistic Features") 
plt.ylabel("Jensen–Shannon Divergence") 
plt.title("Performance of NLI models based on Linguistic Features")
plt.legend() 
plt.show() 
