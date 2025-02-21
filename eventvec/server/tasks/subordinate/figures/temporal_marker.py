

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['dct is _ sub', 'dct is _ matrix', 'matrix is _ sub']


# temporal marker

gpt_no_marker = [.583, .771, .654]
gpt_yesterday = [.569, .763, .624]
gpt_today = [.566, .768, .623]
gpt_now = [.566, .747, .603]
gpt_tomorrow = [.57, .747, .607]
gpt_everyday = [.56, .753, .62]

llama_no_marker = [.596, .748, .757]
llama_yesterday = [.646, .759, .719]
llama_today = [.623, .755, .689]
llama_now = [.589, .697, .679]
llama_tomorrow = [.622, .749, .701]
llama_everyday = [.599, .708, .689]


X_axis = np.arange(len(X)) 

fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
middle = 0.01
width = 0.06
#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

ax.bar(X_axis - (6 * width) - middle, gpt_no_marker, width=width, color='C0', align='center', hatch='//', label = 'gpt_no_marker')
ax.bar(X_axis - (5 * width) - middle, gpt_yesterday, width=width, color='C1', align='center', hatch='//', label = 'gpt_yesterday')
ax.bar(X_axis - (4 * width) - middle, gpt_tomorrow, width=width, color='C4', align='center', hatch='//', label = 'gpt_tomorrow')
ax.bar(X_axis - (3 * width) - middle, gpt_today, width=width, color='C2', align='center', hatch='//', label = 'gpt_today')
ax.bar(X_axis - (2 * width) - middle, gpt_now, width=width, color='C3', align='center', hatch='//', label = 'gpt_now')
ax.bar(X_axis - (1 * width) - middle, gpt_everyday, width=width, color='C5', align='center', hatch='//', label = 'gpt_everyday')

ax.bar(X_axis + (1 * width) + middle, llama_no_marker, width=width, color='C0', align='center', label = 'llama_no_marker')
ax.bar(X_axis + (2 * width) + middle, llama_yesterday, width=width, color='C1', align='center', label = 'llama_yesterday')
ax.bar(X_axis + (3 * width) + middle, llama_tomorrow, width=width, color='C4', align='center', label = 'llama_tomorrow')
ax.bar(X_axis + (4 * width) + middle, llama_today, width=width, color='C2', align='center', label = 'llama_today')
ax.bar(X_axis + (5 * width) + middle, llama_now, width=width, color='C3', align='center', label = 'llama_now')
ax.bar(X_axis + (6 * width) + middle, llama_everyday, width=width, color='C5', align='center', label = 'llama_everyday')


ax = plt.gca()
ax.set_ylim([0.23, 1])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=0) 
plt.xlabel("Temporal Marker") 
plt.ylabel("Model Macro-F1 scores") 
plt.title("Temporal Marker")
plt.legend( loc='upper center', ncol=3) 

plt.savefig('/home/lalady6977/Downloads/temporal_marker.png', bbox_inches='tight')
