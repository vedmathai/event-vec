from eventvec.server.data.wikipedia.datahandlers.wiki_datahandler import WikiDatahandler
from eventvec.server.data.nyt.nyt_datahandlers.nyt_datahandler import NYTDatahandler
from eventvec.server.featurizers.factuality_categorizer.factuality_categorizer import FactualityCategorizer
from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer



if __name__ == '__main__':
    wd = WikiDatahandler()
    files = wd.wiki_file_list()
    factuality_categorizer = FactualityCategorizer()
    linguistic_featurizer = LinguisticFeaturizer()



    print(files)
    for file in files:
        article = wd.read_file(file)
        for line in article:
            if len(line) > 10:
                fdoc = linguistic_featurizer.featurize_document(line.lower())
                for sent in fdoc.sentences():
                    verbs = []
                    for token in sent.tokens():
                        if token.pos() in ['VERB']:
                            verbs += [token.text()]
                            category = factuality_categorizer.categorize(sent.text(), token.text())

                            if category.is_subordinate_of_if():
                                print()
                                print(token.text(), sent.text())
                    if 'if' in sent.text().split():
                        print()
                        print(verbs)
                        print(sent.text())