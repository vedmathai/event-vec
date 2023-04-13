
after_sentence = '{} happened after {}.'
before_sentence = '{} happened before {}.'
in_sentence = '{} happened in {}.'

def create_dates_document():
    sentences = []
    for date_1 in range(1900, 2050):
        in_between = []
        in_sentence_instance = in_sentence.format(date_1, date_1)
        for date_2 in range(date_1 + 1, 2051):
            after_sentence_instance = after_sentence.format(date_2, date_1)
            before_sentence_instance = before_sentence.format(date_1, date_2)
            in_between.append(in_sentence_instance)
            in_between.append(after_sentence_instance)
            in_between.append(before_sentence_instance)
        sentences.append(' '.join(in_between))
    return sentences
