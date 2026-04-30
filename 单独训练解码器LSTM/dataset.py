import os
import json
import torch
from torch.utils.data import Dataset
import re


class MoleculeDataset(Dataset):
    """
    Dataset for conditional SMILES generation with
    - Morgan fingerprint (binary)
    - SMILES sequence
    - Heavy atom composition derived from molecular formula (DiffMS-style)

    Each sample corresponds to ONE molecule.
    """

    # ==================================================
    # Debug configuration (可随时关闭)
    # ==================================================
    DEBUG_DUMP = True          # ← 调试阶段 True，正式训练 False
    DEBUG_MAX = 5              # ← 每种最多输出多少条

    DEBUG_DIR = r"D:\实验\数据\DSSTox_Feb_2024\Intermediate running part results"

    DEBUG_TOKEN_PATH = os.path.join(
        DEBUG_DIR, "debug_tokenization.jsonl"
    )

    DEBUG_FORMULA_PATH = os.path.join(
        DEBUG_DIR, "debug_formula_parsing.jsonl"
    )

    def __init__(
        self,
        data_list,
        tokenizer,
        fp_dim: int = 1024,
        heavy_atoms=("C", "N", "O", "S", "P", "F", "Cl", "Br", "I"),
        max_length: int = 128,
    ):
        """
        Args:
            data_list: list of dicts, each dict contains:
                {
                    "smiles": str,
                    "fingerprint": array-like (fp_dim,),
                    "formula": str   # e.g. "C12H10N2O"
                }
            tokenizer: SMILES tokenizer
            fp_dim: dimension of fingerprint
            heavy_atoms: elements to be counted (no H)
            max_length: max SMILES length
        """
        self.data = data_list
        self.tokenizer = tokenizer
        self.fp_dim = fp_dim
        self.heavy_atoms = list(heavy_atoms)
        self.max_length = max_length

        # debug counters
        self._debug_token_count = 0
        self._debug_formula_count = 0

        # create debug directory once
        if self.DEBUG_DUMP:
            os.makedirs(self.DEBUG_DIR, exist_ok=True)

    def __len__(self):
        return len(self.data)

    def _parse_formula(self, formula: str):
        """
        Parse molecular formula into heavy atom counts.

        Example:
            "C12H10N2O" -> {"C":12, "N":2, "O":1}
        """
        tokens = re.findall(r"([A-Z][a-z]*)(\d*)", formula)

        atom_counts = {}
        for atom, count in tokens:
            if atom == "H":
                continue  # explicitly ignore hydrogen
            atom_counts[atom] = int(count) if count else 1

        counts = [
            atom_counts.get(atom, 0)
            for atom in self.heavy_atoms
        ]

        return torch.tensor(counts, dtype=torch.float)

    def __getitem__(self, idx):
        item = self.data[idx]

        smiles = item["smiles"]
        fingerprint = item["fingerprint"]
        formula = item["formula"]

        # --------------------------------------------------
        # 1. SMILES -> token ids
        # --------------------------------------------------
        ids = self.tokenizer.encode(
            smiles,
            add_sos=True,
            add_eos=True,
        )  # List[int]

        # truncate
        ids = ids[: self.max_length]

        # padding
        pad_id = self.tokenizer.token_to_id[self.tokenizer.PAD_TOKEN]
        if len(ids) < self.max_length:
            ids = ids + [pad_id] * (self.max_length - len(ids))

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)

        # --------------------------------------------------
        # DEBUG: dump tokenization (只输出前 DEBUG_MAX 条)
        # --------------------------------------------------
        if (
            self.DEBUG_DUMP
            and self._debug_token_count < self.DEBUG_MAX
        ):
            debug_record = {
                "idx": idx,
                "smiles": smiles,
                "token_ids_full": ids,                 # 含 SOS / EOS / PAD
                "input_ids": input_ids.tolist(),       # 实际送入 LSTM
                "target_ids": target_ids.tolist(),
                "max_length": self.max_length,
            }

            with open(self.DEBUG_TOKEN_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(debug_record, ensure_ascii=False) + "\n")

            self._debug_token_count += 1

        # --------------------------------------------------
        # 2. Fingerprint
        # --------------------------------------------------
        fingerprint = torch.tensor(fingerprint, dtype=torch.float)

        # --------------------------------------------------
        # 3. Heavy atom counts (from molecular formula)
        # --------------------------------------------------
        heavy_atom_counts = self._parse_formula(formula)

        # --------------------------------------------------
        # DEBUG: dump formula parsing
        # --------------------------------------------------
        if (
            self.DEBUG_DUMP
            and self._debug_formula_count < self.DEBUG_MAX
        ):
            debug_record = {
                "idx": idx,
                "formula": formula,
                "heavy_atoms_order": self.heavy_atoms,
                "heavy_atom_counts": heavy_atom_counts.tolist(),
            }

            with open(self.DEBUG_FORMULA_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(debug_record, ensure_ascii=False) + "\n")

            self._debug_formula_count += 1

        return {
            "input_ids": input_ids,                  # (T,)
            "target_ids": target_ids,                # (T,)
            "fingerprint": fingerprint,              # (fp_dim,)
            "heavy_atom_counts": heavy_atom_counts,  # (K,)
        }
