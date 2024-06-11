

import numpy as np  
import matplotlib.pyplot as plt  

from matplotlib import rc
#plt.rcParams["font.family"] = "Times New Roman"
  
order = [
    'modal_adjective',
    'sub_of_if',
    'sub_of_belief',
    'negation',
    'is_belief',
    'has_modal',
    'is_speech',
    'sub_of_speech',
    'infinitive_sub',
    'sub_said_has_modal',
]

X = ['Is\nSpeech', 'Modal\nAdjective', 'Is\nbelief', 'Sub of\nBelief', 'Sub of\nSpeech', 
      'Has\nModal', 'Negation', 'Sub of\nIf', 'Infinitive\nSub', 'Sub of \nSaid and\n Has Modal']

labels = {
    'is_speech': 'Is\nSpeech',
    'modal_adjective':'Modal\nAdjective',
    'is_belief': 'Is\nbelief',
    'sub_of_belief': 'Sub of\nBelief',
    'sub_of_speech': 'Sub of\nSpeech',
    'has_modal': 'Has\nModal',
    'negation': 'Negation',
    'sub_of_if': 'Sub of\nIf',
    'infinitive_sub': 'Infinitive\nSub',
    'sub_said_has_modal': 'Sub of \nSaid and\n Has Modal'}

books = [16.8, 10.2, 2, 1.6]
wiki = {'is_speech': 4.5, 'modal_adjective': .45, 'is_belief': 1.6, 'sub_of_belief': 3.6, 'sub_of_speech': 8, 'has_modal': 15.9, 'negation': 1.9, 'sub_of_if': .5, 'infinitive_sub': .5, 'sub_said_has_modal': 1.2}
news = {'is_speech': 23, 'modal_adjective': 2.5, 'is_belief': 5.7, 'sub_of_belief': 14.1, 'sub_of_speech': 66.7, 'has_modal': 65.4, 'negation': 9.4, 'sub_of_if': 6, 'infinitive_sub': 3.6, 'sub_said_has_modal': 16}
finance = {'is_speech': 11.6, 'modal_adjective': 2.8, 'is_belief': 16, 'sub_of_belief': 53.7, 'sub_of_speech': 23.3, 'has_modal': 84.9, 'negation': 8.88, 'sub_of_if': 7.5, 'infinitive_sub': 3.6, 'sub_said_has_modal': 11.17}
mnli = {'is_speech': 9.6, 'modal_adjective': 2.1, 'is_belief': 7.2, 'sub_of_belief': 18.4, 'sub_of_speech': 18.7, 'has_modal': 59.4, 'negation': 14, 'sub_of_if': 7.31, 'infinitive_sub': 1.9, 'sub_said_has_modal': 5.82}

news_list = [23, 2.5, 5.7, 14.1, 66.7, 65.4, 9.4, 6, 3.6, 16]
govt_list = [39.2, 18.3, 5.3, 3.8]
finance_list = [11.6, 2.8, 16, 53.7, 23.3, 84.9, 8.88, 7.5, 3.6, 11.17]
social_list = [21, 12.7, 2.59, 3.06]
mnli_list = [9.6, 2.1, 7.2, 18.4, 18.7, 59.4, 14, 7.31, 1.9, 5.82]

news = [news[i] for i in order]
finance = [finance[i] for i in order]
mnli = [mnli[i] for i in order]
wiki = [wiki[i] for i in order]
X = [labels[i] for i in order]
print(X)


patterns = [ "/" , "\\" , "o" , "*" , "x" , ".", "-", "O", ".", "*" ]


X_axis = np.arange(len(X)) 
width = 0.2

def logify(x):
    return x
    return [max(np.log(i),0) for i in x]
 
plt.plot(X_axis, logify(wiki), 'bs', label = 'wiki', markersize=10, linestyle='')
plt.plot(X_axis, logify(news), 'gv', label = 'news', ms=10, linestyle='')
plt.plot(X_axis, logify(finance), 'r*', label = 'finance', ms=10, linestyle='')
plt.plot(X_axis, logify(mnli), 'kd', label = 'mnli', ms=10, linestyle='-')

#plt.bar(X_axis + 0 * width, govt, width, label = 'govt', hatch=patterns[3])
#plt.bar(X_axis + 2 * width, social, width, label = 'social', hatch=patterns[5])
#plt.bar(X_axis - 2 * width, books, width, label = 'books', hatch=patterns[1])

plt.grid()
plt.xticks(X_axis, X, rotation=45) 
plt.xlabel("Credence-informing Linguistic Features") 
plt.ylabel("Ratio as a percentage of sentences") 
plt.legend()
plt.savefig('/home/lalady6977/Downloads/modals_ratio.png', bbox_inches='tight')



"""
Wiki:
is_belief_act 0.016709832901670982
has_modal 0.15919840801591983
is_negated 0.01934980650193498
has_negation_words 0.01789982100178998
is_speech_act 0.04538954610453896
is_subordinate_of_believe 0.03690963090369096
is_subordinate_of_said 0.08053919460805392
is_subordinate_of_then 0.006239937600623993
is_subordinate_of_if 0.005999940000599994
is_subordinate_of_expects 0.008279917200827991
is_infinive_sub_neg 0.0025899741002589974
has_modal_adjective 0.0045899541004589955
has_modal_adverb 9.99990000099999e-06
has_modal_is_subordinate_of_said 0.012299877001229987


News:
is_subordinate_of_said 0.6675033249667504
has_modal 0.6545934540654593
is_speech_act 0.23013769862301378
is_subordinate_of_then 0.01132988670113299
is_negated 0.09477905220947791
has_negation_words 0.06664933350666494
is_belief_act 0.05755942440575594
is_subordinate_of_believe 0.14180858191418086
is_subordinate_of_if 0.06035939640603594
is_subordinate_of_expects 0.03625963740362596
has_modal_adjective 0.025919740802591976
is_infinive_sub_neg 0.007309926900730993
has_modal_adverb 2.999970000299997e-05
has_modal_is_subordinate_of_said 0.16774832251677482


finance:
is_subordinate_of_if 0.07507924920750793
has_modal 0.8496615033849662
is_subordinate_of_then 0.024589754102458974
is_belief_act 0.16012839871601284
is_subordinate_of_believe 0.5378446215537844
is_subordinate_of_said 0.23340766592334078
is_negated 0.08896911030889691
has_negation_words 0.04132958670413296
is_subordinate_of_expects 0.036439635603643966
is_speech_act 0.11642883571164288
has_modal_adjective 0.02814971850281497
is_infinive_sub_neg 0.0004099959000409996
has_modal_adverb 0.0002899971000289997
has_modal_is_subordinate_of_said 0.11176888231117689

MNLI
is_subordinate_of_if 0.07312926870731293
has_modal 0.5940140598594014
is_belief_act 0.07222927770722293
is_subordinate_of_believe 0.18453815461845383
is_negated 0.14040859591404087
is_speech_act 0.0967690323096769
is_subordinate_of_said 0.1879081209187908
has_negation_words 0.07455925440745592
has_modal_adjective 0.02106978930210698
is_subordinate_of_expects 0.01978980210197898
is_infinive_sub_neg 0.0036599634003659965
is_subordinate_of_then 0.012619873801261988
has_modal_adverb 6.999930000699993e-05
has_modal_is_subordinate_of_said 0.05820941790582094



"""