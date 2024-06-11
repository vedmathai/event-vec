

import numpy as np  
import matplotlib.pyplot as plt  
  
X = ['0-0.749', '0.749-\n0.934', '0.934-\n1.058', '1.058-\n1.58' ]

# 5th percentile bucket

random = [1.009, 1.085, 1.105, 1.208]
random_winner = [1.183, 1.186, 1.213, 1.218]
winner = [.92, 1.003, 1.064, 1.088]
uniform = [1.098, 1.098, 1.098, 1.098]

plain = [0.873, .955, 1.012, 1.064]
rules = [0.903, 0.978, 1.033, 1.17]
roberta_large = [0.855, .933, 1.009, 1.065]

X_axis = np.arange(len(X)) 
  
plt.plot(X_axis,  random, 'ks', label = 'Random', linestyle='--', alpha=0.3)
plt.plot(X_axis,  random_winner, 'k<', label = 'Random-winner-takes-all', linestyle='--', alpha=0.3)
plt.plot(X_axis,  winner, 'kv', label = 'Winner-takes-all', linestyle='--', alpha=0.3)
plt.plot(X_axis,  uniform, 'k*', label = 'Uniform', linestyle='--', alpha=0.3)


plt.plot(X_axis,  plain, 'bs', label = 'RN', linestyle='-')
plt.plot(X_axis, rules, 'gv', label = 'RC + RN', linestyle='-')
plt.plot(X_axis, roberta_large,  'r*', label = 'DC + RN', linestyle='-')

ax = plt.gca()
ax.set_ylim([0.8, 1.55])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=45) 
plt.xlabel("Human Judgement Entropy Buckets") 
plt.ylabel("Cross-Entropy between Human and Model Judgements") 
#plt.title("Cross-entropy between Human and Model Confidences")
plt.legend(loc='upper left') 
plt.savefig('/home/lalady6977/Downloads/cross-entropy.png', bbox_inches='tight')
