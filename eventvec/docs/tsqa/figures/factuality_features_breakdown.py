

import numpy as np  
import matplotlib.pyplot as plt  
  
X = ['Is\nSpeech', 'Modal\nAdjective', 'Is\nbelief', 'Sub of\nBelief', 'Sub of\nSpeech', 
      'Has\nModal', 'Negation', 'Sub of\nIf', 'Infinitive Sub']

# 5th percentile bucket
rules = [0.16, .65, .71, .75, .76, .78, 1.13, .69, .72]
roberta_small = [0.104, 0.147, 0.194, 0.43, 0.403, 0.476, 0.46, 0.919, 1.11]
roberta_large = [0.11, .24, .27, .349, .411, .45, .99, 1.36, 1.08]
distilbert = [0.09, .26, .32, .33, .46, .411, .85, .54, .97]

X_axis = np.arange(len(X)) 
  
plt.plot(X,  rules, 'gv', label = 'Rules', linestyle='--')
plt.plot(X, roberta_small, 'bs', label = 'RoBERTa-base', linestyle='--')
plt.plot(X, roberta_large,  'r*', label = 'RoBERTa-large', linestyle='--')
plt.plot(X,  distilbert, 'kD', label = 'DistilBERT', linestyle='--')


ax = plt.gca()
#ax.set_ylim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=45) 
plt.xlabel("Linguistic Features") 
plt.ylabel("Mean Absolute Error") 
#plt.title("Performance of Credence estimation based on Linguistic Features")
plt.legend() 
plt.savefig('/home/lalady6977/Downloads/factuality_features_breakdown.png', bbox_inches='tight')
