

import numpy as np  
import matplotlib.pyplot as plt  
  
X = ['0-0.264','0.264-0.528','0.528-0.792','0.792-1.056'] 
modals = [0.527, 0.274, 0.157, 0.04] 
without = [0.60, 0.243, 0.125, 0.027] 
  
X_axis = np.arange(len(X)) 
  
plt.bar(X_axis - 0.2, without, 0.4, label = 'Without Modals, speech/belief verbs') 
plt.bar(X_axis + 0.2, modals, 0.4, label = 'With Modals, speech/belief verbs') 
  
plt.xticks(X_axis, X) 
plt.xlabel("Entropies") 
plt.ylabel("Number of MNLI pairs") 
plt.title("Comparing MNLI pairs with and without modals, speech/belief verbs") 
plt.legend() 
plt.show() 


