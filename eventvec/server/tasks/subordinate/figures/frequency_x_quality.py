

import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib
import matplotlib.patches as patches

  
font = {'size'   : 14}
matplotlib.rc('font', **font)

names = {
    1: "Past Simple Past Simple",
    2: "Past Simple Past Perf",
    3: "Past Simple Past Cont",
    4: "Past Simple Pres Simple",
    5: "Past Simple Pres Perf",
    6: "Past Simple Pres Cont",
    7: "Past Simple Future Simple",
    8: "Past Simple Future Cont",
    9: "Pres Simple Past Simple",
    10: "Pres Simple Pres Simple",
    11: "Pres Simple Pres Perf",
    12: "Pres Simple Pres Cont",
    13: "Pres Simple Future Simple",
    14: "Pres Perf Past Simple",
    15: "Pres Perf Pres Cont",
    16: "Future Simple Past Simple",
    17: "Future Simple Pres Simple",
    18: "Future Simple Future Simple",
}


performance = {
    1: .721, 2:.947, 3: .604, 4: .533, 5: .820, 6: .616,
    7: .655, 8: .418, 9: .819, 10: .732, 11: .870, 12: .861,
    13: .654, 14: .731, 15: .705, 16: .491, 17: .420, 18: .727
}

frequency_news = {
    1: 27.4, 2: 1.5, 3: 1, 4: 14.3, 5: 12.6, 6: 11.4, 7: 9.5,
    8: 1.4, 9: 3.9, 10: 3.9, 11: 0.4, 12: 2.6, 13: 1.9, 14: 0.3,
    15 :0.2, 16: 0.8, 17: 0.4, 18: 0.4
}

frequency_books = {
    1: 19.4, 2: 0.7, 3: 1.1, 4: 19.1, 5: 0.7, 6: 0.7, 7: 6.3,
    8: 1.4, 9: 4.2, 10: 5.3, 11: 0.3, 12: 2.3, 13: 1.9, 14: 0.3,
    15 :0.1, 16: 1.5, 17: 1.9, 18: 0.8
}

frequency_wiki = {
    1: 27.4, 2: 1.4, 3: 0.4, 4: 7.2, 5: 0.6, 6: 9.7, 7: 5.5,
    8: 0.8, 9: 9.8, 10: 7.5, 11: 0.1, 12: 4.6, 13: 2.4, 14: 1.6,
    15 :1, 16: 1.2, 17: 0.5, 18: 0.4
}

frequency_speech = {
    1: 8.1, 2: 4.5, 3: 0.1, 4: 3.4, 5: 0.4, 6: 2.4, 7: 4.2,
    8: 0.1, 9: 11.1, 10: 13.2, 11: 1.9, 12: 6.2, 13: 8.3, 14: 0.8,
    15 :0.4, 16: 3, 17: 2.6, 18: 2
}

frequency = frequency_speech


X_axis = np.arange(len(performance)) 
X = X_axis
# matrix_tense



fig, ax = plt.subplots(layout='constrained')

markersize=12
#matplotlib.rcParams.update({'font.size': 20})
middle = 0.0
width = 0.05
X_axis = range(1, 19)

items = [(int(performance[i] * 5), performance[i], frequency_speech[i], frequency_books[i], frequency_news[i], frequency_wiki[i], names[i]) for i in performance.keys()]
sorted_items = sorted(items, key=lambda x: (x[0], x[2]),  reverse=True)
frequencies_speech = [i[2] for i in sorted_items]
frequencies_books = [i[3] for i in sorted_items]
frequencies_news = [i[4] for i in sorted_items]
frequencies_wiki = [i[5] for i in sorted_items]
unembedded = [.85 for i in sorted_items]

performances = [i[1] for i in sorted_items]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(X_axis,  unembedded, 'black', label = 'unembedded clauses\n peformance(baseline)', linestyle='--', alpha=1)

ax1.plot(X_axis,  performances, 'r*', label = 'performance', linestyle='-')
ax2.plot(X_axis,  frequencies_speech, 'C0', label = 'frequency_speech', linestyle='-')
#ax2.plot(X_axis,  frequencies_books, 'C1', label = 'frequency_books', linestyle='--', alpha=.3)
ax2.plot(X_axis,  frequencies_news, 'C2', label = 'frequency_news', linestyle='--', alpha=.3)

#ax2.plot(X_axis,  frequencies_wiki, 'C3', label = 'frequency_wiki', linestyle='--', alpha=.3)

rect = patches.Rectangle((5.5, 0.18), 3.7, .65, linewidth=1, edgecolor='red', facecolor='none', linestyle='dashed')
ax1.add_patch(rect)
rect = patches.Rectangle((9.5, 0.04), 9, .75, linewidth=1, edgecolor='blue', facecolor='none', linestyle='dashed')
ax1.add_patch(rect)



ax1.set_ylim([0, 1.1])
#ax.set_xlim([0.2, 0.5])


X_axis_names = [i[6] for i in sorted_items]
ax1.set_xticks(X_axis, X_axis_names, rotation=90) 
ax1.set_xlabel("Relationship") 
ax1.set_ylabel("Model Macro-F1 scores") 
ax2.set_ylabel('Frequency as a %')
#plt.title("Macro-F1 scores of the models grouped\nby Human Judgement Entropy Buckets")
ax1.legend( loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2) 
ax2.legend( loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2) 



plt.savefig('/home/lalady6977/Downloads/frequency_x_performance.png', bbox_inches='tight')
