

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['dct is _ sub', 'dct is _ matrix', 'matrix is _ sub']

# is_quote

gpt_yes = [.489, .74, .627]
gpt_no= [.562, .753, .622]
llama_yes = [.525, .714, .73]
llama_no = [.6, .708, .69]

X_axis = np.arange(len(X)) 

fig, ax = plt.subplots(layout='constrained')

markersize=16
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=22) 

#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

ax.bar(X_axis-0.2, gpt_yes, width=0.1, color='C0', align='center', hatch='//', label = 'gpt_yes')
ax.bar(X_axis-0.1, gpt_no, width=0.1, color='C1', align='center', hatch='//', label = 'gpt_no')

ax.bar(X_axis+0.1, llama_yes, width=0.1, color='C0', align='center', label = 'llama_yes')
ax.bar(X_axis+0.2, llama_no, width=0.1, color='C1', align='center', label = 'llama_no')


ax = plt.gca()
ax.set_ylim([0.45, .9])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=0) 
plt.xlabel("Sub is a quote") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', ncol=3) 

plt.savefig('/home/lalady6977/Downloads/is_quote.png', bbox_inches='tight')
