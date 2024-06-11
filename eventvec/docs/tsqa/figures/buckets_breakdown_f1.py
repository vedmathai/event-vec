

import numpy as np  
import matplotlib.pyplot as plt  
  
X = ['0-0.749', '0.749-\n0.934', '0.934-\n1.058', '1.058-\n1.58' ]

# 5th percentile bucket

plain = [0.701, .625, .573, 0.529]
rules = [0.706, 0.632, .562, .531]
distilbert = [0.715, .674, .602, .543]
llama_3 = [.825, .776, .73, .543]
llama_3_factuality = [.83, .79, .73, .634]

X_axis = np.arange(len(X)) 
  
plt.plot(X_axis,  plain, 'r*', label = 'RM', linestyle='-')
plt.plot(X_axis, rules, 'r*', label = 'RC + RM', linestyle='-.')
plt.plot(X_axis, distilbert,  'r*', label = 'DC + RM', linestyle='--')
plt.plot(X_axis, llama_3,  'cD', label = 'LM', linestyle='-')
plt.plot(X_axis, llama_3_factuality,  'cD', label = 'LC + LM', linestyle='--')



ax = plt.gca()
ax.set_ylim([0.52, 1])
#ax.set_xlim([0.2, 0.5])

plt.grid()
plt.xticks(X_axis, X, rotation=45) 
plt.xlabel("Human Judgement Entropy Buckets") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend(loc='upper left') 
plt.savefig('/home/lalady6977/Downloads/split_f1.png', bbox_inches='tight')
