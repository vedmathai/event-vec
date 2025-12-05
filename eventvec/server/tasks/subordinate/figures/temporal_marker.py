

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.patches as patches
import matplotlib

font = {'size'   : 13}
matplotlib.rc('font', **font)
  
X = ['no marker', 'yesterday', 'today', 'tomorrow', 'now', 'everyday']

# matrix_tense

"""

gpt_oss_120b_past = [0.754, 0.898, 0.785, 0.784, 0.407, 0.733]
gpt_oss_120b_present= [.701, .552, .684, .656, .654, .618]
gpt_oss_120b_future = [.632, .552, .569, .804, .459, .618]

deepseek_past = [.744, .872, .729, .660, .456, .720]
deepseek_present = [.592, .715, .590, .872, .616, .505]
deepseek_future = [.556, .677, .556, .757, .407, .485]

"""
gpt_oss_120b_past = [.640, .750, .660, .692, .447, .602]
gpt_oss_120b_present= [.790, .698, .807, .819, .601, .741]
gpt_oss_120b_future = [.656, .554, .571, .729, .466, .625]

deepseek_past = [.595, .823, .637, .743, .456, .514]
deepseek_present = [.776, .823, .724, .830, .658, .727]
deepseek_future = [.519, .618, .514, .717, .358, .470]

gpt_5_past = [.816, .857, .685, .719, .619, .667]
gpt_5_present = [.824, .739, .816, .886, .667, .793]
gpt_5_future = [.662, .603, .523, .514, .537, .500]


X_axis = np.arange(len(X))

fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
middle = 0.0
width = 0.05
#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

ax.plot(X_axis, gpt_oss_120b_past, color='C0', linestyle='solid', label = 'gpt_oss_120b_past', alpha=1)
ax.plot(X_axis, gpt_oss_120b_present, color='C0', linestyle='dotted', label = 'gpt_oss_120b_present', alpha=1)
ax.plot(X_axis, gpt_oss_120b_future, color='C0', linestyle='dashed', label = 'gpt_oss_120b_future', alpha=1)

ax.plot(X_axis, gpt_5_past, color='C1', linestyle='solid', label = 'gpt_5_past', alpha=1)
ax.plot(X_axis, gpt_5_present, color='C1', linestyle='dotted', label = 'gpt_5_present', alpha=1)
ax.plot(X_axis, gpt_5_future, color='C1', linestyle='dashed', label = 'gpt_5_future', alpha=1)

"""

ax.plot(X_axis, llama_70_past, color='C2', linestyle='solid', label = 'llama_70_past', alpha=1)
ax.plot(X_axis, llama_70_present, color='C2', linestyle='dotted', label = 'llama_70_present', alpha=1)
ax.plot(X_axis, llama_70_future, color='C2', linestyle='dashed', label = 'llama_70_future', alpha=1)

ax.plot(X_axis, llama_8_past, color='C3', linestyle='solid', label = 'llama_8_past', alpha=1)
ax.plot(X_axis, llama_8_present, color='C3', linestyle='dotted', label = 'llama_8_present', alpha=1)
ax.plot(X_axis, llama_8_future, color='C3', linestyle='dashed', label = 'llama_8_future', alpha=1)
"""
ax.plot(X_axis, deepseek_past, color='C4', linestyle='solid', label = 'deepseek_past', alpha=1)
ax.plot(X_axis, deepseek_present, color='C4', linestyle='dotted', label = 'deepseek_present', alpha=1)
ax.plot(X_axis, deepseek_future, color='C4', linestyle='dashed', label = 'deepseek_future', alpha=1)


ax = plt.gca()
ax.set_ylim([0.3, 1])
#ax.set_xlim([0.2, 0.5])

rect = patches.Rectangle((0.75, 0.65), .45, .25, linewidth=1, edgecolor='red', facecolor='none', linestyle='dashed')
matplotlib.pyplot.text(.76, .65, 'R1')
ax.add_patch(rect)
rect = patches.Rectangle((2.75, 0.65), .45, .25, linewidth=1, edgecolor='red', facecolor='none', linestyle='dashed')
ax.add_patch(rect)
matplotlib.pyplot.text(2.76, .65, 'R2')
rect = patches.Rectangle((3.75, 0.35), .45, .35, linewidth=1, edgecolor='red', facecolor='none', linestyle='dashed')
ax.add_patch(rect)
matplotlib.pyplot.text(3.76, .35, 'R3')




plt.xticks(X_axis, X, rotation=0) 
plt.xlabel("Relationship") 
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.15), ncol=2) 


plt.savefig('/home/lalady6977/Downloads/temporal_marker_matrix_tense.png', bbox_inches='tight')
