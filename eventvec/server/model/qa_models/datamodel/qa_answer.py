class QAAnswer:
    def __init__(self):
        self._text = None
        self._paragraph = None
        self._start_location = None
        self._end_location = None

    def text(self):
        return self._text

    def paragraph(self):
        return self._paragraph

    def start_location(self):
        return self._start_location

    def end_location(self):
        return self._end_location

    def set_text(self, text):
        self._text = text

    def set_paragraph(self, paragraph):
        self._paragraph = paragraph

    def set_start_location(self, start_location):
        self._start_location = start_location

    def set_end_location(self, end_location):
        self._end_location = end_location
    
    def to_dict(self):
        return {
            "text": self.text(),
            "paragraph": self.paragraph(),
            "start_location": self.start_location(),
            "end_location": self.end_location(),
        }

    @staticmethod
    def from_dict(val):
        qa_answer_location = QAAnswer()
        qa_answer_location.set_text(val['text'])
        qa_answer_location.set_paragraph(val['paragraph'])
        qa_answer_location.set_start_location(val['start_location'])
        qa_answer_location.set_end_location(val['end_location'])
        return qa_answer_location
