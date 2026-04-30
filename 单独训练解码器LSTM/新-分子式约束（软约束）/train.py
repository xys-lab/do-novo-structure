import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    train_dataset,
    val_dataset,
    tokenizer,
    decoder,
    device="cuda",
    batch_size=64,
    num_epochs=20,
    lr=1e-4,
    lambda_atom=0.1,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    print("========== Entering training loop ==========")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Batch size:    {batch_size}")
    print(f"Epochs:        {num_epochs}")
    print("============================================")

    # =================================================
    # DataLoader
    # =================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # =================================================
    # Model to device
    # =================================================
    decoder.to(device)

    # =================================================
    # optimizer
    # =================================================
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=lr,
    )

    ce_loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id
    )
    mse_loss_fn = nn.MSELoss()

    # =================================================
    # Training loop
    # =================================================
    for epoch in range(num_epochs):
        decoder.train()
        total_train_loss = 0.0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [train]"
        )

        for batch in train_pbar:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            cond = batch["cond_vec"].to(device)
            atom_gt = batch["heavy_atom_counts"].to(device)

            token_logits, atom_pred = decoder(
                input_ids=input_ids,
                cond=cond,
            )

            ce_loss = ce_loss_fn(
                token_logits.reshape(-1, token_logits.size(-1)),
                target_ids.reshape(-1),
            )
            atom_loss = mse_loss_fn(atom_pred, atom_gt)

            loss = ce_loss + lambda_atom * atom_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            train_pbar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        avg_train_loss = total_train_loss / len(train_loader)

        # =================================================
        # Validation
        # =================================================
        decoder.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                cond = batch["cond_vec"].to(device)
                atom_gt = batch["heavy_atom_counts"].to(device)

                token_logits, atom_pred = decoder(
                    input_ids=input_ids,
                    cond=cond,
                )

                ce_loss = ce_loss_fn(
                    token_logits.reshape(-1, token_logits.size(-1)),
                    target_ids.reshape(-1),
                )
                atom_loss = mse_loss_fn(atom_pred, atom_gt)

                loss = ce_loss + lambda_atom * atom_loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"[Epoch {epoch + 1}/{num_epochs}] "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}"
        )