import re
from collections import Counter

class Tokenizer:
    SPECIAL_TOKENS = ["<|unk|>", "<|endoftext|>"]

    def __init__(self, pattern):
        self.regex = re.compile(pattern)
        self.token_to_id = {}
        self.id_to_token = {}

    def build_vocab(self, dataset):
        counter = Counter()

        for row in dataset:
            counter.update(re.findall(self.regex, row['text']))

        vocab_tokens = self.SPECIAL_TOKENS + sorted(counter.keys())
        self.token_to_id = {tok: idx for idx, tok in enumerate(vocab_tokens)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

    def encode(self, text):
        unk_id = self.token_to_id["<|unk|>"]
        tokens = re.findall(self.regex, text)
        return [self.token_to_id.get(tok, unk_id) for tok in tokens]

    def encode_row(self,row):
        row['token_ids'] = self.encode(row['text'])
        return row

    def decode(self, ids):
        return "".join(self.id_to_token[i] for i in ids)

    @property
    def vocab_size(self):
        return len(self.token_to_id)