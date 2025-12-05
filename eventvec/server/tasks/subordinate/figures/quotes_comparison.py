
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
import matplotlib.patches as mpatches

  
X = ['gpt-oss\n-120b', 'GPT-5', 'Llama-8b', 'Llama-70B', 'DeepSeek\n-R1', 'Qwen3']

# matrix_tense

font = {'size'   : 14}
matplotlib.rc('font', **font)

temporal_direct = [.566, .698, .441, .626,  .605, .498]
temporal_direct_replaced = [0.719, .687, .501, .720, .672, .582]
temporal_indirect = [.674, .683, .451,  .706, .669, .544]
temporal_indirect_replaced = [.691, .714, .463, .691, .610, .494]


pronoun_direct = [.642, .920, .294, .569, .789, .66]
pronoun_direct_replaced = [.746, 1, .402, .748, .781, .966]
pronoun_indirect = [.718, .747, .410, .742, .518, .559]
pronoun_indirect_replaced = [.718, 1, .532, .742, .734, .984]

X_axis = np.arange(len(X)) 

fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
middle = 0.0
width = 0.16
#plt.plot(X_axis,  plain_llama, 'r*', label = 'plain_llama', linestyle='-')

ax.bar(X_axis - (1.6 * width) - middle, temporal_direct, width=width, color='C0', align='center', hatch='\\\\', label = 'temporal direct', alpha=1)
ax.bar(X_axis - (1.6 * width) - middle, temporal_direct_replaced, width=width, color='C0', align='center', hatch='\\\\', label = 'temporal direct replaced', alpha=0.2)
ax.bar(X_axis - (0.6 * width) - middle, temporal_indirect, width=width, color='C1', align='center', hatch='\\\\', label = 'temporal indirect', alpha=1)
ax.bar(X_axis - (0.6 * width) - middle, temporal_indirect_replaced, width=width, color='C1', align='center', hatch='\\\\', label = 'temporal indirect replaced', alpha=0.2)



ax.bar(X_axis + (0.6 * width) - middle, pronoun_direct, width=width, color='C2', align='center', hatch='//', label = 'pronoun direct', alpha=1)
ax.bar(X_axis + (0.6 * width) - middle, pronoun_direct_replaced, width=width, color='C2', align='center', hatch='//', label = 'pronoun direct replaced', alpha=0.2)
ax.bar(X_axis + (1.6 * width) - middle, pronoun_indirect, width=width, color='C3', align='center', hatch='//', label = 'pronoun indirect', alpha=1)
ax.bar(X_axis + (1.6 * width) - middle, pronoun_indirect_replaced, width=width, color='C3', align='center', hatch='//', label = 'pronoun indirect replaced', alpha=0.2)


ax = plt.gca()
ax.set_ylim([0.2, 1.05])
#ax.set_xlim([0.2, 0.5])

plt.xlabel("Models")
plt.xticks(X_axis, X, rotation=20)
plt.ylabel("Model Macro-F1 scores") 
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
#plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.35), ncol=3) 

red_patch = mpatches.Patch(color='C0', hatch='\\\\', label='Temporal Direct')
blue_patch = mpatches.Patch(color='C1', hatch='\\\\', label='Temporal Indirect')
green_patch = mpatches.Patch(color='C2', hatch='//', label='Pronoun Direct')
purple_patch = mpatches.Patch(color='C3', hatch='//', label='Pronoun Indirect')
white_patch = mpatches.Patch(color='C0', alpha=0.4, label='(Lower opacity:\nexchanged labels)')


plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.35), ncol=2, handles=[red_patch, blue_patch, green_patch, purple_patch, white_patch])


plt.savefig('/home/lalady6977/Downloads/quotes_comparison.png', bbox_inches='tight')

