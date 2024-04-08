

import numpy as np  
import matplotlib.pyplot as plt  
  
X = ['Auxiliary\nto\nSentences', 'Embedded\nVerbs\nto Sentences', 'Embedded\nAuxilaries\nto sentences', 'Adverbs\nto\nSentences']

books = [16.8, 10.2, 2, 1.6]
wiki = [4.4, 2.3, 0.2, 0.4]
news = [19.6, 30.1, 5.3, 1.9]
govt = [39.2, 18.3, 5.3, 3.8]
finance = [25.2, 32.5, 5.03, 5.4]
social = [21, 12.7, 2.59, 3.06]
mnli = [16.8, 9.16, 1.98, 2.1]

patterns = [ "/" , "\\" , "o" , "*" , "x" , ".", "-", "O", ".", "*" ]

  
X_axis = np.arange(len(X)) 
width = 0.12
  
plt.bar(X_axis - 3 * width, wiki, width, label = 'wiki', hatch=patterns[0])
plt.bar(X_axis - 2 * width, books, width, label = 'books', hatch=patterns[1])
plt.bar(X_axis - 1 * width, news, width, label = 'news', hatch=patterns[2])
plt.bar(X_axis + 0 * width, govt, width, label = 'govt', hatch=patterns[3])
plt.bar(X_axis + 1 * width, finance, width, label = 'finance', hatch=patterns[4])
plt.bar(X_axis + 2 * width, social, width, label = 'social', hatch=patterns[5])
plt.bar(X_axis + 3 * width, mnli, width, label = 'mnli', hatch=patterns[6])


plt.xticks(X_axis, X) 
plt.xlabel("Genres of the sources of naturistic text") 
plt.ylabel("Ratio as a percentage") 
plt.title("Comparing frequencies of occurrence of\ndifferent phenomena in various sources of naturistic text") 
plt.legend() 
plt.show() 
