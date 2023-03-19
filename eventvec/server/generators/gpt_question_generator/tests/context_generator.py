import unittest

from eventvec.server.config import Config
from eventvec.server.data_handlers.qa_datahandlers.tsqa_datahandler.tsqa_datahandler import TSQADatahandler
from eventvec.server.model.qa_models.datamodel.qa_dataset import QADataset
from eventvec.server.generators.gpt_question_generator.gpt_question_generator import GPTQuestionGenerator


class TSQA2QA_test(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config.instance()

    def test_tsqa2qa_converter(self):
        prompts_templates = [
            "{}", "",
            #"Paraphrasing {}",
            #"In order for {}, he had to do "
        ]
        tsqa_datahandler = TSQADatahandler()
        qa_data = tsqa_datahandler.qa_data()
        data = qa_data.data()
        generator = GPTQuestionGenerator()
        generated_sentences = []
        for datum in data:
            context = datum.context()
            for answer in datum.answers():
                paragraph = context[answer.paragraph_idx()]
                i = answer.start_location()
                while i >= 0 and paragraph[i] not in '.?!':
                    i -= 1
                j = answer.end_location()
                while j < len(paragraph) and paragraph[j] not in '.?,':
                    j+=1
                sentence = paragraph[i+1:j]
                for template in prompts_templates:
                    q = template.format(sentence)
                    generated_sentence = generator.generate(q)
                    generated_sentences.extend(generated_sentence)




if __name__ == '__main__':
    unittest.main()