import unittest

from eventvec.server.config import Config
from eventvec.server.generators.gpt_question_generator.gpt_question_generator import GPTQuestionGenerator


class TSQANoiseGenerator:
    def __init__(self) -> None:
        self.config = Config.instance()
        self._generator = GPTQuestionGenerator()

    def generate(self, context, start_char, end_char):
        context = context.split()
        prompts_templates = [
            ("{}", ""),
            #("Paraphrase: ", ""),
            #("In order for {}, he had to do ", "He had to do ")
        ]
        generated_contexts = []
        i = start_char
        while i >= 0 and not any(k in context[i] for k in '.?,'):
            i -= 1
        j = end_char
        while j < len(context) and not any(k in context[j] for k in '.?,'):
            j+=1
        sentence = context[i+1:j]
        if len(sentence) == 0:
            sentence = context
        context_prefix = context[:i+1]
        context_suffix = context[j:]
        for template, suffix in prompts_templates:
            q = template.format(' '.join(sentence))
            generated_sentence = self._generator.generate(q)
            generated_part = self._get_sentence(len(q), generated_sentence)
            generated_sentence = context_prefix + [suffix] + [generated_part] + context_suffix
            generated_contexts.append(' '.join(generated_sentence))
        return generated_contexts

    def _get_sentence(self, start, generated_sentence):
        i = start
        while i < len(generated_sentence) -1 and generated_sentence[i] != '.':
            i += 1
        return generated_sentence[start: i + 1]        