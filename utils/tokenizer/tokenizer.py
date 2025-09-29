import torch
import os
import json
import re
from collections import Counter

class Tokenizer:
    def __init__(self, vocab_size=50000, max_length=512):
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.PAD_ID = 0
        self.SOS_ID = 1
        self.EOS_ID = 2
        self.UNK_ID = 3

        self.ZERO_ID = 4
        self.ONE_ID = 5
        self.TWO_ID = 6
        self.THREE_ID = 7
        self.FOUR_ID = 8
        self.FIVE_ID = 9
        self.SIX_ID = 10
        self.SEVEN_ID = 11
        self.EIGHT_ID = 12
        self.NINE_ID = 13

        self.a_ID = 14
        self.b_ID = 15
        self.c_ID = 16
        self.d_ID = 17
        self.e_ID = 18
        self.f_ID = 19
        self.g_ID = 20
        self.h_ID = 21
        self.i_ID = 22
        self.j_ID = 23
        self.k_ID = 24
        self.l_ID = 25
        self.m_ID = 26
        self.n_ID = 27
        self.o_ID = 28
        self.p_ID = 29
        self.q_ID = 30
        self.r_ID = 31
        self.s_ID = 32
        self.t_ID = 33
        self.u_ID = 34
        self.v_ID = 35
        self.w_ID = 36
        self.x_ID = 37
        self.y_ID = 38
        self.z_ID = 39


        self.special_tokens = {
            "<PAD>": self.PAD_ID,
            "<SOS>": self.SOS_ID,
            "<EOS>": self.EOS_ID,
            "<UNK>": self.UNK_ID,
            "0": self.ZERO_ID,
            "1": self.ONE_ID,
            "2": self.TWO_ID,
            "3": self.THREE_ID,
            "4": self.FOUR_ID,
            "5": self.FIVE_ID,
            "6": self.SIX_ID,
            "7": self.SEVEN_ID,
            "8": self.EIGHT_ID,
            "9": self.NINE_ID,
            "a": self.a_ID,
            "b": self.b_ID,
            "c": self.c_ID,
            "d": self.d_ID,
            "e": self.e_ID,
            "f": self.f_ID,
            "g": self.g_ID,
            "h": self.h_ID,
            "i": self.i_ID,
            "j": self.j_ID,
            "k": self.k_ID,
            "l": self.l_ID,
            "m": self.m_ID,
            "n": self.n_ID,
            "o": self.o_ID,
            "p": self.p_ID,
            "q": self.q_ID,
            "r": self.r_ID,
            "s": self.s_ID,
            "t": self.t_ID,
            "u": self.u_ID,
            "v": self.v_ID,
            "w": self.w_ID,
            "x": self.x_ID,
            "y": self.y_ID,
            "z": self.z_ID
        }

        self.token2id = {}
        self.id2token = {}

    def _tokenize(self, text):
        text = text.lower()
        return re.findall(r'\d|[a-z]+|[^\w\s]', text)

    def update_dicts(self, counter: Counter):
        # in case stuff like <UNK> is in the training text and counter
        for token in self.special_tokens:
            counter.pop(token, None)

        self.token2id = dict(self.special_tokens)

        most_common = counter.most_common(self.vocab_size - len(self.special_tokens))
        for idx, (token, _) in enumerate(most_common):
            token_id = idx + len(self.special_tokens)
            self.token2id[token] = token_id
        self.id2token = {v: k for k, v in self.token2id.items()}

    def encode(self, text, add_SOS=False, add_EOS=False, pad_cut=False):
        tokens = self._tokenize(text)
        ids = []

        if add_SOS:
            ids.append(self.SOS_ID)

        for token in tokens:
            if token not in self.token2id and token.isalpha():
                for letter in token:
                    ids.append(self.token2id.get(letter))
                continue
            ids.append(self.token2id.get(token, self.UNK_ID))

        if pad_cut:
            if len(ids) < self.max_length:
                padding = [self.PAD_ID] * (self.max_length - len(ids))
                ids.extend(padding)

        if pad_cut and add_EOS:
            ids = ids[:self.max_length-1]
            ids.append(self.EOS_ID)
        elif pad_cut:
            ids = ids[:self.max_length]
        elif add_EOS:
            ids.append(self.EOS_ID)

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, x):
        # Flatten the tensor and decode each element
        flat_batch = x.flatten()
        return [self.id2token.get(int(idx.item())) for idx in flat_batch]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "token2id": self.token2id,
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            max_length=data["max_length"]
        )
        tokenizer.token2id = data["token2id"]
        tokenizer.id2token = {v: k for k, v in tokenizer.token2id.items()}
        return tokenizer
