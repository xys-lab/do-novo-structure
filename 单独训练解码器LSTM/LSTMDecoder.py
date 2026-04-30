import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMDecoder(nn.Module):
    """
    Conditional LSTM Decoder with Heavy Atom Prediction Head
    -------------------------------------------------------
    Safe version: force NON-cuDNN LSTM for new GPUs (sm_120)
    """

    def __init__(
        self,
        vocab_size: int,
        cond_dim: int = 1024,
        emb_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_heavy_atoms: int = 9,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heavy_atoms = num_heavy_atoms

        # 1. Token embedding
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # 2. Condition -> initial LSTM state
        self.h0_proj = nn.Linear(cond_dim, hidden_dim)
        self.c0_proj = nn.Linear(cond_dim, hidden_dim)

        # 3. LSTM core
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 4. Token prediction head
        self.token_head = nn.Linear(hidden_dim, vocab_size)

        # 5. Heavy atom prediction head
        self.atom_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heavy_atoms),
        )

        self.dropout = nn.Dropout(dropout)

    def _init_state(self, cond):
        """
        cond: (B, cond_dim)
        returns:
            h0, c0: (num_layers, B, hidden_dim)
        """
        B = cond.size(0)

        h0 = self.h0_proj(cond)
        c0 = self.c0_proj(cond)

        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.num_layers, 1, 1)

        return h0.contiguous(), c0.contiguous()

    def forward(
            self,
            input_ids,
            cond,
            return_atom_counts: bool = True,
    ):
        B, T = input_ids.size()

        # 1. Embedding
        emb = self.embedding(input_ids)
        emb = self.dropout(emb)

        # 2. Init state
        h, c = self._init_state(cond)

        # 3. LSTM forward
        if emb.is_cuda:
            # GPU 上：临时关闭 cuDNN，避免 sm_120 问题
            with torch.backends.cudnn.flags(enabled=False):
                out, (h_n, c_n) = self.lstm(emb, (h, c))
        else:
            # CPU 上：正常跑
            out, (h_n, c_n) = self.lstm(emb, (h, c))

        # 4. Token prediction
        token_logits = self.token_head(out[:, :-1, :])

        if not return_atom_counts:
            return token_logits

        # 5. Heavy atom prediction
        final_hidden = h_n[-1]
        atom_pred = self.atom_head(final_hidden)

        return token_logits, atom_pred
