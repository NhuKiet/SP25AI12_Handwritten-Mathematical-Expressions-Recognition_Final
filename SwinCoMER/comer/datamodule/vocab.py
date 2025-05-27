import os
from functools import lru_cache
from typing import Dict, List

@lru_cache()
def default_dict():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionary.txt")

class LatexVocab:
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    def __init__(self, dict_path: str = default_dict()) -> None:
        self.word2idx = dict()
        self.word2idx["<pad>"] = self.PAD_IDX
        self.word2idx["<sos>"] = self.SOS_IDX
        self.word2idx["<eos>"] = self.EOS_IDX

        with open(dict_path, "r") as f:
            for line in f.readlines():
                w = line.strip()
                if w:
                    self.word2idx[w] = len(self.word2idx)

        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}

        print(f"Init vocab with size: {len(self.word2idx)}")
        print(f"Vocab keys: {list(self.word2idx.keys())}")

    def words2indices(self, words: List[str]) -> List[int]:
        indices = []
        for w in words:
            if w in self.word2idx:
                indices.append(self.word2idx[w])
            else:
                print(f"Warning: Word '{w}' not in vocabulary, skipping")
                indices.append(self.PAD_IDX)
        return indices

    def indices2words(self, id_list: List[int]) -> List[str]:
        return [self.idx2word[i] for i in id_list if i in self.idx2word]

    def indices2label(self, id_list: List[int]) -> str:
        words = self.indices2words(id_list)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

vocab = LatexVocab()