

import numpy as np  
import matplotlib.pyplot as plt  
  
X = ['Government','Finance','Reddit','News','Books', 'Wiki', 'MNLI']
modals = [39, 25.2, 21, 19, 16.8, 4.4, 16.8]
speech = [36.4, 28, 18.4, 16.8, 28.7, 8.5, 20]

  
X_axis = np.arange(len(X)) 
  
plt.bar(X_axis - 0.2, modals, 0.4, label = 'Modals to Sentences')
plt.bar(X_axis + 0.2, speech, 0.4, label = 'Speech-act to Sentences') 
  
plt.xticks(X_axis, X) 
plt.xlabel("Genres of the sources of naturistic text") 
plt.ylabel("Ratio as a percentage") 
plt.title("Comparing the ratio of modals and speech-act verbs to sentences in different genres of naturistic text") 
plt.legend() 
plt.show() 
