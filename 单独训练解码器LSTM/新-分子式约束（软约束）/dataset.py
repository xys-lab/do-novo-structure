import os
import json
import torch
from torch.utils.data import Dataset
import re


class MoleculeDataset(Dataset):
    """
    Dataset for conditional SMILES generation with
    - 1024-d MLP embedding
    - 22-d MS condition
    - SMILES sequence
    - Heavy atom composition derived from molecular formula
    """

    # ==================================================
    # Debug configuration
    # ==================================================
    DEBUG_DUMP = True
    DEBUG_MAX = 5

    DEBUG_DIR = r"D:\实验\数据\DSSTox_Feb_2024\Intermediate running part results"

    DEBUG_TOKEN_PATH = os.path.join(
        DEBUG_DIR, "debug_tokenization.jsonl"
    )

    DEBUG_FORMULA_PATH = os.path.join(
        DEBUG_DIR, "debug_formula_parsing.jsonl"
    )

    DEBUG_COND_PATH = os.path.join(
        DEBUG_DIR, "debug_cond_vec.jsonl"
    )

    def __init__(
        self,
        data_list,
        tokenizer,
        cond_dim: int = 1046,
        heavy_atoms=("C", "N", "O", "S", "P", "F", "Cl", "Br", "I"),
        max_length: int = 128,
    ):
        self.data = data_list
        self.tokenizer = tokenizer
        self.cond_dim = cond_dim
        self.heavy_atoms = list(heavy_atoms)
        self.max_length = max_length

        self._debug_token_count = 0
        self._debug_formula_count = 0
        self._debug_cond_count = 0

        if self.DEBUG_DUMP:
            os.makedirs(self.DEBUG_DIR, exist_ok=True)

    def __len__(self):
        return len(self.data)

    def _parse_formula(self, formula: str):
        """
        Parse molecular formula into heavy atom counts.
        Example:
            C12H10N2O -> [12,2,1,...]
        """
        if not isinstance(formula, str):
            formula = ""

        tokens = re.findall(r"([A-Z][a-z]*)(\d*)", formula)

        atom_counts = {}
        for atom, count in tokens:
            if atom == "H":
                continue
            atom_counts[atom] = int(count) if count else 1

        counts = [atom_counts.get(atom, 0) for atom in self.heavy_atoms]
        return torch.tensor(counts, dtype=torch.float)

    def __getitem__(self, idx):
        item = self.data[idx]

        smiles = item["smiles"]
        formula = item["formula"]
        mlp_vec = item["mlp_vec"]
        ms_vec = item["ms_vec"]

        # --------------------------------------------------
        # 1. SMILES -> token ids
        # --------------------------------------------------
        ids = self.tokenizer.encode(
            smiles,
            add_sos=True,
            add_eos=True,
        )

        ids = ids[: self.max_length]

        pad_id = self.tokenizer.token_to_id[self.tokenizer.PAD_TOKEN]
        if len(ids) < self.max_length:
            ids = ids + [pad_id] * (self.max_length - len(ids))

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)

        if self.DEBUG_DUMP and self._debug_token_count < self.DEBUG_MAX:
            debug_record = {
                "idx": idx,
                "smiles": smiles,
                "token_ids_full": ids,
                "input_ids": input_ids.tolist(),
                "target_ids": target_ids.tolist(),
                "max_length": self.max_length,
            }

            with open(self.DEBUG_TOKEN_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(debug_record, ensure_ascii=False) + "\n")

            self._debug_token_count += 1

        # --------------------------------------------------
        # 2. Build cond_vec = [mlp_vec, ms_vec]
        # --------------------------------------------------
        cond_vec = mlp_vec + ms_vec
        cond_vec = torch.tensor(cond_vec, dtype=torch.float)

        if cond_vec.numel() != self.cond_dim:
            raise ValueError(
                f"cond_vec 维度错误，期望 {self.cond_dim}，实际 {cond_vec.numel()}"
            )

        if self.DEBUG_DUMP and self._debug_cond_count < self.DEBUG_MAX:
            debug_record = {
                "idx": idx,
                "mlp_dim": len(mlp_vec),
                "ms_dim": len(ms_vec),
                "cond_dim": int(cond_vec.numel()),
            }

            with open(self.DEBUG_COND_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(debug_record, ensure_ascii=False) + "\n")

            self._debug_cond_count += 1

        # --------------------------------------------------
        # 3. Heavy atom counts (from formula)
        # --------------------------------------------------
        heavy_atom_counts = self._parse_formula(formula)

        if self.DEBUG_DUMP and self._debug_formula_count < self.DEBUG_MAX:
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
            "input_ids": input_ids,
            "target_ids": target_ids,
            "cond_vec": cond_vec,
            "heavy_atom_counts": heavy_atom_counts,
        }