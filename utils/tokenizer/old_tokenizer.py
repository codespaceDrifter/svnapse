import numpy as np
import torch
import os
import json
import re
from collections import Counter

'''
tokenizer format:
english, digits, puncutation only
all letters lower case
white spaces are part of tokens



tokenizing Trie:
a Trie is a nested dict of chars built from known tokens  
there should be a END_OF_TOKEN key that takes to the id of the sequence that led to there  
when encoding you walk that 24(max) length in the trie keeping a running longest_id_found. i.e. known tokens: cat, catch, car  
{
  'c': {
    'a': {
      't': {
        'END_OF_TOKEN': 10,    # "cat" is token ID 10
        'c': {
          'h': {
            'END_OF_TOKEN': 11  # "catch" is token ID 11
          }
        }
      },
      'r': {
        'END_OF_TOKEN': 12      # "car" is token ID 12
      }
    }
  }
  'END_OF_TOKEN': 0
}
when matching 'catch' you keep walking, first longest_id_found is 0, then at CAT it's 10, and then at catch it's 11.



learning counting dict
start with some special tokens that are single chars. including digits, lower case letters, white space, and punctuation
then use vocab_mix_from_sources.py to generate a txt from all the training txt for a representative vocabulary
do the following in a loop until vocab size is reached:
1: tokenize the mixed.txt into a numpy using sel
2: use counter to count pairs. for example if the ids are of 'c' 'a' 't' then it is ('c','a') pair +1 and ('a','t') pair +1. 
note that if the two tokens plused together exceed max_token_length they are not added. if one of them is a special character i.e. UNK they are not added.
3: find the top k pairs. k could be like 500 or something. add the top k pairs to self.token_to_id with ascending ids. rebuild id_to_token. call build_trie
'''



class Tokenizer:
    def __init__(self, vocab_size=32000, max_token_length=24):
        self.vocab_size = vocab_size
        self.max_token_length = max_token_length 

        self.PAD_ID = 0
        self.SOS_ID = 1
        self.EOS_ID = 2
        self.UNK_ID = 3

        self.special_tokens = {
            "<PAD>": self.PAD_ID,
            "<SOS>": self.SOS_ID,
            "<EOS>": self.EOS_ID,
            "<UNK>": self.UNK_ID,
        }

        self.starting_tokens = {
            # Digits
            '0': 4, '1': 5, '2': 6, '3': 7, '4': 8, '5': 9, '6': 10, '7': 11, '8': 12, '9': 13,
            # Lowercase letters
            'a': 14, 'b': 15, 'c': 16, 'd': 17, 'e': 18, 'f': 19, 'g': 20, 'h': 21, 'i': 22, 
            'j': 23, 'k': 24, 'l': 25, 'm': 26, 'n': 27, 'o': 28, 'p': 29, 'q': 30, 'r': 31, 
            's': 32, 't': 33, 'u': 34, 'v': 35, 'w': 36, 'x': 37, 'y': 38, 'z': 39,
            # Whitespace
            ' ': 40, '\n': 41, '\t': 42,
            # Punctuation
            '.': 43, ',': 44, '!': 45, '?': 46, ';': 47, ':': 48, '-': 49, '_': 50, 
            '(': 51, ')': 52, '[': 53, ']': 54, '{': 55, '}': 56, '"': 57, "'": 58, 
            '/': 59, '\\': 60, '|': 61, '@': 62, '#': 63, '$': 64, '%': 65, '^': 66, 
            '&': 67, '*': 68, '+': 69, '=': 70, '<': 71, '>': 72, '~': 73, '`': 74
        }

        self.token_to_id = {**self.special_tokens, **self.starting_tokens}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        self.trie = {}
        self._build_trie()

    def _build_trie(self):
        self.trie = {}
        for token, id in self.token_to_id.items():
            if token in self.special_tokens: 
                continue
            node = self.trie
            for char in token:
                if char not in node:
                    node[char] = {}
                node = node [char]
            node["END_OF_TOKEN"] = id

    def tokenize(self, text):
        text = text.lower()
        i = 0
        id_list = []
        text_length = len(text)
        while i < text_length:
            max_walk_length = min(self.max_token_length, len(text)-i)
            j = 0
            node = self.trie
            longest_token_id = self.UNK_ID
            while j < max_walk_length and text[i+j] in node:
                node = node[text[i+j]]
                if 'END_OF_TOKEN' in node:
                    longest_token_id = node['END_OF_TOKEN']
                j += 1
            id_list.append(longest_token_id)

            i = i + max(j,1) # if j first round not in trie, an UNK is still added but j = 0 which will not advance i which will cause permanent loop so we max(j,1)
        return np.array(id_list, dtype=np.int32)

    # writes a bin file
    # later open with memmap np.dtype = 32
    def parse_file(self, input_file_path, output_file_path, chunk_size=10_000_000):
        file_size = os.path.getsize(input_file_path)

        with open(output_file_path, "wb") as fout:  # raw binary file
            with open(input_file_path, "r", encoding="utf-8", errors="ignore") as fin:
                # the := opereator assigns chunk and checks "if chunk"(check if it's not empty) at the same time
                while chunk := fin.read(chunk_size):
                    token_ids = self.tokenize(chunk)
                    token_ids.tofile(fout)
                    pos = fin.tell()
                    print(f"parsing file chunk {pos / file_size * 100:.3f}%")


    # if k is too big then the parallel merged tokens get too long too quickly
    def _fit_vocab_source(self, txt_path, k = 500):
        bin_path = txt_path.replace('.txt', '.bin')
        checkpoint_path = os.path.join(os.path.dirname(txt_path),'tokenizer_checkpoint.json')
        while len(self.token_to_id) < self.vocab_size:
            # parse file into memmap
            self.parse_file(txt_path,bin_path)
            token_ids = np.memmap(bin_path, dtype=np.int32, mode = 'r')
            pair_counts = {}
            # count adjacent pairs for future token making
            for i in range(len(token_ids) - 1):
                id1, id2 = token_ids[i], token_ids[i+1]
                #if any of them are special tokens, continue
                if id1 < 4 or id2 < 4:
                    continue

                #if the two combined together have length > self.max_token_length, continue
                token1,token2 = self.id_to_token[id1], self.id_to_token[id2]
                if len(token1) + len(token2) > self.max_token_length:
                    continue



                pair = (id1,id2)
                pair_counts[pair] = pair_counts.get(pair,0) + 1

                if (i % 1_000_000 == 0):
                    print ('current pair merged', self.id_to_token[id1] + self.id_to_token[id2])
                    print (f"counting pairs {i / len(token_ids) * 100 :.3f}%")
            del token_ids
            # sort pairs
            sorted_pair_counts = sorted(pair_counts.items(), key = lambda x:x[1] , reverse  =True)
            # add merged tokens. use a added counter to make sure skips do not affect how much added per round
            added = 0
            for (id1, id2), _ in sorted_pair_counts:
                if len(self.token_to_id) >= self.vocab_size:
                    break
                if added >= k:
                    break
                merged_token = self.id_to_token[id1] + self.id_to_token[id2]
                # skip duplicated merges: (a, bc) and (ab, c) could both be added in top pairs
                if merged_token in self.token_to_id:
                    continue
                self.token_to_id[merged_token] = len(self.token_to_id)
                added += 1

            # temp checkpoint save to not lose all progress
            self.save(checkpoint_path)

            print (f'building tokenizer {len(self.token_to_id)} / {self.vocab_size} vocab')

            #rebuild tokenizer
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
            self._build_trie()


    def save(self, path):
        d = os.path.dirname(path)
        if d: #maybe it's just a file name with no folder prefix
            os.makedirs(d, exist_ok=True)

        data = {
            "vocab_size": self.vocab_size,
            "max_token_length": self.max_token_length,
            "token_to_id": self.token_to_id,
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            max_token_length=data["max_token_length"]
        )
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
        tokenizer._build_trie()
        return tokenizer

