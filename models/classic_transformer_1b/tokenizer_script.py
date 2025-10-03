from utils import tokenizer
from utils.tokenizer.tokenizer import *

# python3 -m models.classic_transformer_1b.tokenizer_script

tokenizer = Tokenizer(vocab_size=1000)

tokenizer._fit_vocab_source(txt_path='models/classic_transformer_1b/tokenizer/data/mixed_vocab_data.txt')

tokenizer.save('models/classic_transformer_1b/tokenizer/tokenizer.json')
