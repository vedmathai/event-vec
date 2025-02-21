

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['no marker', 'yesterday', 'today', 'now', 'tomorrow', 'everyday']


# temporal marker

past = [.832, .802, .741, .698, .675, .688]
present = [.819, .734, .696, .698, .676, .695]
future = [.757, .719, .689, .679, .701, .689]



X_axis = np.arange(len(X)) 

fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
width = 0.2
#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

ax.bar(X_axis - (1 * width), past, width=width, color='C0', align='center', label = 'past')
ax.bar(X_axis - (0 * width), present, width=width, color='C1', align='center', label = 'present')
ax.bar(X_axis - (-1 * width), future, width=width, color='C4', align='center', label = 'future')


ax = plt.gca()
ax.set_ylim([0.23, 1])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=0) 
plt.xlabel("Temporal Marker") 
plt.ylabel("Model Macro-F1 scores") 
plt.title("Temporal Marker split by subordinate tense for GPT-4o")
plt.legend( loc='upper center', ncol=3) 

plt.savefig('/home/lalady6977/Downloads/temporal_marker_x_sub_tense.png', bbox_inches='tight')
