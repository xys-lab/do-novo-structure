import os
import json
import torch
import torch.nn as nn


class FingerprintProjector(nn.Module):
    """
    Project discrete Morgan fingerprint (0/1, 1024-d)
    into continuous conditional embedding (1024-d)
    """

    # ==================================================
    # Debug configuration (与 dataset.py 同等级)
    # ==================================================
    DEBUG_DUMP = True        # ← 调试 True，正式训练 False
    DEBUG_MAX = 5

    DEBUG_DIR = r"D:\实验\数据\DSSTox_Feb_2024\Intermediate running part results"
    DEBUG_PATH = os.path.join(
        DEBUG_DIR, "debug_projector_cond.jsonl"
    )

    def __init__(self, fp_dim=1024, cond_dim=1024):
        super().__init__()

        self.fp_dim = fp_dim
        self.cond_dim = cond_dim

        # 简单线性投影（你现在的设计）
        self.proj = nn.Linear(fp_dim, cond_dim)

        # debug counter
        self._debug_count = 0

        if self.DEBUG_DUMP:
            os.makedirs(self.DEBUG_DIR, exist_ok=True)

    def forward(self, fp):
        """
        Args:
            fp: (B, fp_dim), binary {0,1}

        Returns:
            cond: (B, cond_dim), continuous
        """
        cond = self.proj(fp)

        # --------------------------------------------------
        # DEBUG: dump fingerprint -> cond (只前 DEBUG_MAX 条)
        # --------------------------------------------------
        if self.DEBUG_DUMP and self._debug_count < self.DEBUG_MAX:
            batch_size = fp.size(0)

            for i in range(batch_size):
                if self._debug_count >= self.DEBUG_MAX:
                    break

                debug_record = {
                    "global_idx": self._debug_count,
                    "fingerprint_discrete": fp[i].detach().cpu().tolist(),
                    "cond_continuous": cond[i].detach().cpu().tolist(),
                    "fp_dim": self.fp_dim,
                    "cond_dim": self.cond_dim,
                }

                with open(self.DEBUG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(debug_record, ensure_ascii=False) + "\n")

                self._debug_count += 1

        return cond
