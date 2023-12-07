

import numpy as np  
import matplotlib.pyplot as plt  
  
X = ['0-0.663','0.663-0.912','0.912-1.045','1.045-1.206','1.206-1.584'] 
modals = [0.076, 0.157, 0.171, 0.281, 0.312] 
without = [0.107, 0.161, 0.2, 0.26, 0.269] 
  
X_axis = np.arange(len(X)) 
  
plt.bar(X_axis - 0.2, without, 0.4, label = 'Without Modals, speech/belief verbs') 
plt.bar(X_axis + 0.2, modals, 0.4, label = 'With Modals, speech/belief verbs') 
  
plt.xticks(X_axis, X) 
plt.xlabel("Entropies") 
plt.ylabel("Number of MNLI pairs") 
plt.title("Comparing MNLI pairs with and without modals, speech/belief verbs") 
plt.legend() 
plt.show() 
