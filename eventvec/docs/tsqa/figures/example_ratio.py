
import numpy as np
import matplotlib.pyplot as plt
 
N = 4
 
factuality_mnli = (0.424, 0.44, 0.5, 0.544)
factuality_snli = (0.061, 0.036, 0.066, 0.036)

ind = np.arange(N)   
 
fig = plt.subplots(figsize =(10, 7))

plt.plot(ind,  factuality_mnli, 'gs', label = 'Ratio of MNLI with Modals', linestyle='--')
plt.plot(ind,  factuality_snli, 'bo', label = 'Ratio of SNLI with Modals', linestyle='--')

plt.ylabel('Contribution')
plt.title('Ratio of datapoints with credence-informing linguistic\nfeatures for MNLI and SNLI divided by entropy buckets')
plt.xticks(ind, ['0-\n0.749','0.749-\n0.934','0.934-\n1.058','1.058-\n1.58'], rotation=45)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel("Entropy buckets of the human annotation") 
plt.ylabel("Contribution of examples to each bucket") 
#plt.legend((p1[0], p2[0]), ('boys', 'girls'))
plt.legend() 
 
plt.savefig('/home/lalady6977/Downloads/mnli_pairs.png', bbox_inches='tight')