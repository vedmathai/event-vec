import unittest

from eventvec.server.entry_points.vectorizer.coreference_resolver import CoreferenceResolver
from eventvec.server.model.document_models.document_model import Document


class TestCoreferenceResolver(unittest.TestCase):

    def test_coreference_resolver(self):
        resolver = CoreferenceResolver()
        document = Document()
        document.set_text('Matt had been drawing a bird after learning how to. The teacher taught him')
        document = resolver.resolve(document)
        self.assertEqual(
            document.resolved_text(),
            "Matt had been drawing a bird after learning how to . The teacher taught Matt"
        )



if __name__ == '__main__':
    unittest.main()