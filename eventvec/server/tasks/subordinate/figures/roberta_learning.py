

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['DCT-sub', 'DCT-matrix', 'matrix-sub']


# temporal marker

past_greyed = [.723, .818, .786]
present_greyed= [.703, .833, .747]
future_greyed = [.807, .856, .735]

test_past = [.585, .792, .647]
test_present = [.49, .78, .455]
test_future = [.386, .785, .241]
random_split = [.681, .84, .733]
by_sentence = [.723, .823, .724]

X_axis = np.arange(len(X)) 

fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
middle = 0
width = 0.08
#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

ax.bar(X_axis - (2* width) - middle, past_greyed, width=width, color='C0', align='center', hatch='\\', label = 'past ablation', alpha=0.3)
ax.bar(X_axis - (1 * width) - middle, present_greyed, width=width, color='C1', align='center', hatch='\\', label = 'present ablation', alpha=0.3)
ax.bar(X_axis - (0 * width) - middle, future_greyed, width=width, color='C4', align='center', hatch='\\', label = 'future ablation', alpha=0.3)

ax.bar(X_axis - (2* width) - middle, test_past, width=width, color='C0', align='center', hatch='\\', label = 'test_past')
ax.bar(X_axis - (1 * width) - middle, test_present, width=width, color='C1', align='center', hatch='\\', label = 'test_present')
ax.bar(X_axis - (0 * width) - middle, test_future, width=width, color='C4', align='center', hatch='\\', label = 'test_future')
ax.bar(X_axis + (1 * width) - middle, random_split, width=width, color='C2', align='center', hatch='\\', label = 'random_selection')
ax.bar(X_axis + (2 * width) - middle, by_sentence, width=width, color='C3', align='center', hatch='\\', label = 'entire_range')


ax = plt.gca()
ax.set_ylim([0.2, 1])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=0) 
plt.xlabel("Relationship") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("RoBERTa learning properties")
plt.legend( loc='upper center', ncol=3) 

plt.savefig('/home/lalady6977/Downloads/roberta_learning.png', bbox_inches='tight')
