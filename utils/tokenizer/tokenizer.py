# ByteTokenizer - byte-level BPE using HuggingFace tokenizers
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

VOCAB_SIZE = 50176


class ByteTokenizer:
    """
    Byte-level BPE tokenizer.

    Usage:
        # training
        tok = ByteTokenizer.create()
        tok.train(text_iterator)
        tok.save()

        # inference
        tok = ByteTokenizer.load()
        ids = tok.encode("hello world")
        text = tok.decode(ids)
    """

    SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.pad_id = None
        self.unk_id = None
        self.bos_id = None
        self.eos_id = None

        if tokenizer:
            self._load_special_ids()

    def _load_special_ids(self):
        """loads special token ids from vocab"""
        vocab = self.tokenizer.get_vocab()
        self.pad_id = vocab.get("<|pad|>")
        self.unk_id = vocab.get("<|unk|>")
        self.bos_id = vocab.get("<|bos|>")
        self.eos_id = vocab.get("<|eos|>")

    # ---- factory methods ----

    @classmethod
    def create(cls):
        """creates new untrained tokenizer"""
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.post_processor = processors.ByteLevel(trim_offsets=False)
        return cls(tok)

    @classmethod
    def load(cls, path):
        """loads trained tokenizer from json"""
        tok = Tokenizer.from_file(path)
        print(f"loaded tokenizer: {path} (vocab: {tok.get_vocab_size()})")
        return cls(tok)

    # ---- training ----

    def train(self, text_iterator):
        """trains on text iterator"""
        trainer = trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            min_frequency=2,
            special_tokens=self.SPECIAL_TOKENS,
            show_progress=True,
        )
        print(f"training (target vocab: {VOCAB_SIZE})...")
        self.tokenizer.train_from_iterator(text_iterator, trainer)
        self._load_special_ids()
        print(f"done (final vocab: {self.tokenizer.get_vocab_size()})")

    def save(self, path):
        """saves to json"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tokenizer.save(path)
        print(f"saved: {path}")

    # ---- encode/decode ----

    def encode(self, text, add_special_tokens=False):
        """text -> token ids"""
        ids = self.tokenizer.encode(text).ids
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        """token ids -> text"""
        if skip_special_tokens:
            special = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
            ids = [i for i in ids if i not in special]
        return self.tokenizer.decode(ids)

    def __len__(self):
        return self.tokenizer.get_vocab_size()
