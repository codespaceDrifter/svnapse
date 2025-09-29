from collections import Counter
import re
import os
import json

class TokenCounter:
    def __init__(self):
        self.counter = Counter()
    
    def _tokenize(self, text):
        text = text.lower()
        return re.findall(r'\d|[a-z]+|[^\w\s]', text)

    def fit_counter_string(self, text):
        tokens = self._tokenize(text)
        self.counter.update(tokens)

    def fit_counter_txt(self, path, start_line=0, end_line=None):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if start_line is not None and i < start_line:
                    continue
                if end_line is not None and i >= end_line:
                    break
                line = line.strip()
                if line:
                    self.fit_counter_string(line)
                if i % 1000 == 0:
                    print(f"Processed {i} lines...") 

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dict(self.counter), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            counter_dict = json.load(f)
        token_counter = cls()
        token_counter.counter = Counter(counter_dict)
        return token_counter

