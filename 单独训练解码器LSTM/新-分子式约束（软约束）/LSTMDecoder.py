import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    """
    Conditional LSTM decoder
    - cond: [1024-d MLP embedding + 22-d MS] = 1046-d
    - output: token logits + heavy atom prediction
    """

    def __init__(
        self,
        vocab_size,
        cond_dim=1046,
        emb_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.1,
        num_heavy_atoms=9,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.cond_dim = cond_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heavy_atoms = num_heavy_atoms

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
        self.atom_head = nn.Linear(hidden_dim, num_heavy_atoms)

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

    def forward(self, input_ids, cond):
        """
        input_ids: (B, T)
        cond:      (B, cond_dim)

        returns:
            token_logits: (B, T, vocab_size)
            atom_pred:    (B, num_heavy_atoms)
        """
        x = self.token_emb(input_ids)  # (B, T, emb_dim)

        h0, c0 = self._init_state(cond)

        out, (h_n, c_n) = self.lstm(x, (h0, c0))  # out: (B, T, H)

        token_logits = self.token_head(out)  # (B, T, vocab_size)

        final_hidden = h_n[-1]               # (B, H)
        atom_pred = self.atom_head(final_hidden)

        return token_logits, atom_pred