

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
  
X = ['no marker', 'yesterday', 'today', 'tomorrow', 'now', 'everyday']

# matrix_tense

gpt_open_yes = [.8, .750, 0.778, 0.821, 0.614, 0.786]
gpt_open_no= [.766, 0.610, 0.752, 0.729, 0.570, 0.730]

gpt_5_yes = [.769, .610, .717, .695, .543, .631]
gpt_5_no = [.709, .789, .671, .688, .571, .619]

llama_405_yes = [0.809, .825, 0.695, 0.889, 0.688, 0.797]
llama_405_no = [0.750, 0.823, 0.629, 0.693, 0.612, 0.726]

X_axis = np.arange(len(X)) 

fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
middle = 0.0
width = 0.05
#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

#ax.plot(X_axis, gpt_open_yes, color='C0', linestyle='dashed', label = 'gpt_open_yes', alpha=1)
#ax.plot(X_axis, gpt_open_no, color='C0', linestyle='solid', label = 'gpt_open_no', alpha=0.7)

#ax.plot(X_axis, gpt_5_yes, color='C1', linestyle='dashed', label = 'gpt_5_yes', alpha=1)
#ax.plot(X_axis, gpt_5_no, color='C1', linestyle='solid', label = 'gpt_5_no', alpha=0.7)

ax.plot(X_axis, llama_405_yes, color='C2', linestyle='dashed', label = 'llama_70_yes', alpha=1)
ax.plot(X_axis, llama_405_no, color='C2', linestyle='solid', label = 'llama_70_no', alpha=0.7)

ax = plt.gca()
ax.set_ylim([0.4, 0.85])
#ax.set_xlim([0.2, 0.5])

plt.xticks(X_axis, X, rotation=0) 
plt.xlabel("Relationship") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.15)) 


plt.savefig('/home/lalady6977/Downloads/temporal_marker_matrix_sub.png', bbox_inches='tight')
