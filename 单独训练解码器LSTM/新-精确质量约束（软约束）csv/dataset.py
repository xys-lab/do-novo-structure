import os
import json
import torch
from torch.utils.data import Dataset


class MoleculeDataset(Dataset):
    """
    Dataset for conditional SMILES generation with
    - 1024-d MLP embedding
    - 22-d MS condition
    - SMILES sequence
    - exact_mass as auxiliary supervision target
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

    DEBUG_COND_PATH = os.path.join(
        DEBUG_DIR, "debug_cond_vec.jsonl"
    )

    DEBUG_MASS_PATH = os.path.join(
        DEBUG_DIR, "debug_exact_mass.jsonl"
    )

    def __init__(
        self,
        data_list,
        tokenizer,
        cond_dim: int = 1046,
        max_length: int = 128,
    ):
        self.data = data_list
        self.tokenizer = tokenizer
        self.cond_dim = cond_dim
        self.max_length = max_length

        self.pad_id = self.tokenizer.token_to_id[self.tokenizer.PAD_TOKEN]

        self._debug_token_count = 0
        self._debug_cond_count = 0
        self._debug_mass_count = 0

        if self.DEBUG_DUMP:
            os.makedirs(self.DEBUG_DIR, exist_ok=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        smiles = item["smiles"]
        exact_mass = float(item["exact_mass"])
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

        if len(ids) < self.max_length:
            ids = ids + [self.pad_id] * (self.max_length - len(ids))

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)

        # 有效 token mask：padding 位置为 0，真实 token 为 1
        token_mask = (target_ids != self.pad_id).long()

        if self.DEBUG_DUMP and self._debug_token_count < self.DEBUG_MAX:
            debug_record = {
                "idx": idx,
                "smiles": smiles,
                "token_ids_full": ids,
                "input_ids": input_ids.tolist(),
                "target_ids": target_ids.tolist(),
                "token_mask": token_mask.tolist(),
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
        # 3. exact_mass
        # --------------------------------------------------
        exact_mass = torch.tensor(exact_mass, dtype=torch.float)

        if self.DEBUG_DUMP and self._debug_mass_count < self.DEBUG_MAX:
            debug_record = {
                "idx": idx,
                "exact_mass": float(exact_mass.item()),
            }

            with open(self.DEBUG_MASS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(debug_record, ensure_ascii=False) + "\n")

            self._debug_mass_count += 1

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "cond_vec": cond_vec,
            "token_mask": token_mask,
            "exact_mass": exact_mass,
        }