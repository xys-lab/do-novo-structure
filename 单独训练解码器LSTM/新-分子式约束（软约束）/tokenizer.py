# tokenizer.py
import json
import re
from pathlib import Path
from typing import List, Dict, Iterable


class SmilesTokenizer:
    """
    SMILES Tokenizer (rule-based, robust)

    - 支持多数据集统一建词表
    - 词表可保存 / 加载
    - <PAD>, <SOS>, <EOS>, <UNK>
    - 与 Dataset / 路径完全解耦
    """

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab_built = False

    # --------------------------------------------------
    # 1. SMILES rule-based tokenization
    # --------------------------------------------------
    @staticmethod
    def tokenize_smiles(smiles: str) -> List[str]:
        """
        基于通用 SMILES 规则的切分（覆盖 >99% 常见情况）
        """
        pattern = (
            r"\[[^\]]+\]"         # [NH4+], [C@H], [O-]
            r"|Br|Cl|Si|Na|Ca"    # 常见双字符原子
            r"|%\d{2}"            # %12 环编号
            r"|\d"                # 单个数字
            r"|\(|\)|\.|=|#|-|\+|\\|/|:|~|\?|>|<|\*|\$"
            r"|[A-Za-z]"          # 单字符原子
        )
        return re.findall(pattern, smiles)

    # --------------------------------------------------
    # 2. Build vocabulary from SMILES iterable
    # --------------------------------------------------
    def build_vocab(
        self,
        smiles_iter: Iterable[str],
        min_freq: int = 1
    ):
        token_freq: Dict[str, int] = {}

        for smi in smiles_iter:
            if not isinstance(smi, str):
                continue
            for tok in self.tokenize_smiles(smi):
                token_freq[tok] = token_freq.get(tok, 0) + 1

        vocab = [
            self.PAD_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
        ]

        for token, freq in sorted(token_freq.items()):
            if freq >= min_freq:
                vocab.append(token)

        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        self.pad_token_id = self.token_to_id[self.PAD_TOKEN]
        self.sos_token_id = self.token_to_id[self.SOS_TOKEN]
        self.eos_token_id = self.token_to_id[self.EOS_TOKEN]
        self.unk_token_id = self.token_to_id[self.UNK_TOKEN]
        self.vocab_built = True

    # --------------------------------------------------
    # 3. Encode / Decode
    # --------------------------------------------------
    def encode(self, smiles: str, add_sos=True, add_eos=True) -> List[int]:
        assert self.vocab_built, "Tokenizer vocab not built!"

        ids = []
        if add_sos:
            ids.append(self.token_to_id[self.SOS_TOKEN])

        for tok in self.tokenize_smiles(smiles):
            ids.append(self.token_to_id.get(tok, self.token_to_id[self.UNK_TOKEN]))

        if add_eos:
            ids.append(self.token_to_id[self.EOS_TOKEN])

        return ids

    def decode(self, ids: List[int], remove_special=True) -> str:
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, self.UNK_TOKEN)
            if remove_special and tok in {
                self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN
            }:
                continue
            tokens.append(tok)
        return "".join(tokens)

    # --------------------------------------------------
    # 4. Padding
    # --------------------------------------------------
    def pad(self, ids: List[int], max_len: int) -> List[int]:
        pad_id = self.token_to_id[self.PAD_TOKEN]
        return ids[:max_len] + [pad_id] * max(0, max_len - len(ids))

    # --------------------------------------------------
    # 5. Save / Load
    # --------------------------------------------------
    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token_to_id": self.token_to_id}, f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {i: t for t, i in tokenizer.token_to_id.items()}

        tokenizer.pad_token_id = tokenizer.token_to_id[tokenizer.PAD_TOKEN]
        tokenizer.sos_token_id = tokenizer.token_to_id[tokenizer.SOS_TOKEN]
        tokenizer.eos_token_id = tokenizer.token_to_id[tokenizer.EOS_TOKEN]
        tokenizer.unk_token_id = tokenizer.token_to_id[tokenizer.UNK_TOKEN]
        tokenizer.vocab_built = True

        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)


