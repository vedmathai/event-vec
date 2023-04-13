
class TSQAParagraph:
    def __init__(self):
        self._title = None
        self._text = None

    def title(self):
        return self._title

    def text(self):
        return self._text
    
    def set_title(self, title):
        self._title = title

    def set_text(self, text):
        self._text = text

    @staticmethod
    def from_dict(val):
        tsqa_paragraph = TSQAParagraph()
        tsqa_paragraph.set_title(val['title'])
        tsqa_paragraph.set_text(val['text'])
        return tsqa_paragraph