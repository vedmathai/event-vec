
true_positives = {
    'b': 1776,
    'a': 1444,
    'd': 642 + 826 + 147  + 40 + 25 + 45 + 28 + 23 + 31,
}

false_negatives = {
    'b': 17 + 71 + 81 + 12 + 500,
    'a': 22 + 34 + 76 + 8 + 441,
    'd': 88 + 32 + 289 + 37 + 102 + 21 + 9 + 64,
}

false_positives = {
    'b': 22 + 88 + 37 + 21,
    'a': 17 + 32 + 102 + 9,
    'd': 71 + 34 + 81 + 76 + 12 + 8
}

precision = {}
recall = {}
f1 = {}
for token in true_positives:
    precision[token] = float(true_positives[token]) / (true_positives[token] + false_positives[token])
    recall[token] = float(true_positives[token]) / (true_positives[token] + false_negatives[token])
    f1[token] = (2 * precision[token] * recall[token]) / (precision[token] + recall[token])

print('precision', precision)
print('recall', recall)
print('f1', f1)