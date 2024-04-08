from transformers import DistilBertTokenizer
from transformers import DistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained("transformers_cache/distilbert-base-uncased")
model.save_pretrained("transformers_cache/distilbert-base-uncased")


from transformers import AlbertTokenizer
from transformers import AlbertModel
tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v2")
model = AlbertModel.from_pretrained("albert-xxlarge-v2")
tokenizer.save_pretrained("transformers_cache/albert-xxlarge-v2")
model.save_pretrained("transformers_cache/albert-xxlarge-v2")

from transformers import BartTokenizer, BartModel
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartModel.from_pretrained('facebook/bart-large')
tokenizer.save_pretrained("transformers_cache/bart-large")
model.save_pretrained("transformers_cache/bart-large")

from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaModel.from_pretrained('roberta-large')
tokenizer.save_pretrained("transformers_cache/roberta-large")
model.save_pretrained("transformers_cache/roberta-large")

from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
tokenizer.save_pretrained("transformers_cache/roberta-base")
model.save_pretrained("transformers_cache/roberta-base")

from transformers import XLNetTokenizer, XLNetModel
tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
model = XLNetModel.from_pretrained('xlnet-large-cased')
tokenizer.save_pretrained("transformers_cache/xlnet-large-cased")
model.save_pretrained("transformers_cache/xlnet-large-cased")

from transformers import ElectraTokenizer, ElectraModel
tokenizer = ElectraTokenizer.from_pretrained('google/electra-large-discriminator')
model = ElectraModel.from_pretrained('google/electra-large-discriminator')
tokenizer.save_pretrained("transformers_cache/electra-large-discriminator")
model.save_pretrained("transformers_cache/electra-large-discriminator")

# T5
from transformers import T5Tokenizer, T5Model
tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5Model.from_pretrained('t5-large')
tokenizer.save_pretrained("transformers_cache/t5-large")
model.save_pretrained("transformers_cache/t5-large")

# DeBERTa
from transformers import DebertaTokenizer, DebertaModel
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-large')
model = DebertaModel.from_pretrained('microsoft/deberta-large')
tokenizer.save_pretrained("transformers_cache/deberta-large")
model.save_pretrained("transformers_cache/deberta-large")

# XLM-RoBERTa
from transformers import XLMRobertaTokenizer, XLMRobertaModel
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
model = XLMRobertaModel.from_pretrained('xlm-roberta-large')
tokenizer.save_pretrained("transformers_cache/xlm-roberta-large")
model.save_pretrained("transformers_cache/xlm-roberta-large")
