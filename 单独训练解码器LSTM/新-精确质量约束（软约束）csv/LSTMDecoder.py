import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    """
    Conditional LSTM decoder
    - cond: [1024-d MLP embedding + 22-d MS] = 1046-d
    - output: token logits + exact mass prediction
    """

    def __init__(
        self,
        vocab_size,
        cond_dim=1046,
        emb_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.1,
        mass_hidden_dim=128,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.cond_dim = cond_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mass_hidden_dim = mass_hidden_dim

        self.token_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
        )

        self.h0_proj = nn.Linear(cond_dim, num_layers * hidden_dim)
        self.c0_proj = nn.Linear(cond_dim, num_layers * hidden_dim)

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.token_head = nn.Linear(hidden_dim, vocab_size)

        self.mass_head = nn.Sequential(
            nn.Linear(hidden_dim, mass_hidden_dim),
            nn.ReLU(),
            nn.Linear(mass_hidden_dim, 1)
        )

    def _init_state(self, cond):
        """
        cond: (B, cond_dim)
        return:
            h0, c0: (num_layers, B, hidden_dim)
        """
        batch_size = cond.size(0)

        h0 = self.h0_proj(cond).view(
            batch_size, self.num_layers, self.hidden_dim
        ).transpose(0, 1).contiguous()

        c0 = self.c0_proj(cond).view(
            batch_size, self.num_layers, self.hidden_dim
        ).transpose(0, 1).contiguous()

        return h0, c0

    def forward(self, input_ids, cond, token_mask):
        """
        input_ids: (B, T)
        cond:      (B, cond_dim)
        token_mask:(B, T), 真实 token 为 1，padding 为 0

        returns:
            token_logits: (B, T, vocab_size)
            mass_pred:    (B,)
        """
        x = self.token_emb(input_ids)  # (B, T, emb_dim)

        h0, c0 = self._init_state(cond)

        out, _ = self.lstm(x, (h0, c0))   # out: (B, T, H)
        token_logits = self.token_head(out)

        mask = token_mask.unsqueeze(-1).float()   # (B, T, 1)
        masked_out = out * mask
        seq_repr = masked_out.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        mass_pred = self.mass_head(seq_repr).squeeze(-1)

        return token_logits, mass_pred