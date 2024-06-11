

import numpy as np  
import matplotlib.pyplot as plt  
  
X = ['0-0.749', '0.749-\n0.934', '0.934-\n1.059', '1.059-\n1.58' ]

X = [
        'Modal\nAdjective',
        'Sub of\nIf',
        'Sub of\nBelief',
        'Negation',
        'Is\nBelief',
        'Has\nModal',
        'Is\nSpeech',
        'Sub of\nSpeech', 
        'Infinitive Sub'
    ]
# 5th percentile bucket
plain = [0.359, .453, .525, .527, .543, .57, .571, .597, .683]
rules = [0.395, .463, .55, .557, .608, .587, .573, .58, .66]
roberta_large = [.429, .428, .553, .561, .592, .597, .616, .632, .664]
llama_plain = [.357, .5614, .5834, .561, .579, .617, .61, .6144, .6102]
llama_factuality = [.4786, .601, .5836, .5744, .613, .6288, .6238, .6174, .5938 ]

X_axis = np.arange(len(X)) 
  
plt.plot(X,  plain, 'r*', label = 'RM', markersize=10, linestyle='-')
plt.plot(X, rules, 'r*', label = 'RC + RM', markersize=10, linestyle='-.')
plt.plot(X, roberta_large,  'r*', label = 'DC + RM', markersize=10, linestyle='--')
plt.plot(X, llama_plain,  'cD', label = 'LM', linestyle='-')
plt.plot(X, llama_factuality,  'cD', label = 'LC + LM', linestyle='--')


ax = plt.gca()
ax.set_ylim([0.3, 0.7])
plt.xticks(X_axis, X, rotation=45) 
plt.xlabel("Linguistic Features") 
plt.ylabel("F1 scores") 
#plt.title("Performance of ChaosNLI based on Linguistic Features")
plt.legend() 
plt.savefig('/home/lalady6977/Downloads/nli-breakdown-features.png', bbox_inches='tight')
