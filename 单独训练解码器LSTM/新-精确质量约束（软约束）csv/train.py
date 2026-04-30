import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def ppm_quadratic_loss(
    mass_pred,
    mass_true,
    alpha=1e-6,
):
    """
    单一非线性二次 ppm 损失

    计算流程：
    1. 先计算 ppm 误差：
       ppm = |mass_pred - mass_true| / mass_true * 1e6

    2. 再代入单一二次函数：
       loss = alpha * ppm^2

    3. 对一个 batch 取平均
    """
    ppm = torch.abs(mass_pred - mass_true) / torch.clamp(mass_true, min=1e-8) * 1e6
    loss = alpha * (ppm ** 2)
    return loss.mean()


def train(
    train_dataset,
    val_dataset,
    tokenizer,
    decoder,
    device="cuda",
    batch_size=64,
    num_epochs=20,
    lr=1e-4,
    lambda_mass=0.1,
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

    # =================================================
    # Training loop
    # =================================================
    for epoch in range(num_epochs):
        decoder.train()
        total_train_loss = 0.0
        total_train_ce = 0.0
        total_train_mass = 0.0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [train]"
        )

        for batch in train_pbar:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            cond = batch["cond_vec"].to(device)
            token_mask = batch["token_mask"].to(device)
            exact_mass = batch["exact_mass"].to(device)

            token_logits, mass_pred = decoder(
                input_ids=input_ids,
                cond=cond,
                token_mask=token_mask,
            )

            ce_loss = ce_loss_fn(
                token_logits.reshape(-1, token_logits.size(-1)),
                target_ids.reshape(-1),
            )

            mass_loss = ppm_quadratic_loss(
                mass_pred=mass_pred,
                mass_true=exact_mass,
                alpha=1e-6,
            )

            loss = ce_loss + lambda_mass * mass_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_ce += ce_loss.item()
            total_train_mass += mass_loss.item()

            train_pbar.set_postfix(
                total=f"{loss.item():.4f}",
                ce=f"{ce_loss.item():.4f}",
                mass=f"{mass_loss.item():.4f}",
            )

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_ce = total_train_ce / len(train_loader)
        avg_train_mass = total_train_mass / len(train_loader)

        # =================================================
        # Validation
        # =================================================
        decoder.eval()
        total_val_loss = 0.0
        total_val_ce = 0.0
        total_val_mass = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                cond = batch["cond_vec"].to(device)
                token_mask = batch["token_mask"].to(device)
                exact_mass = batch["exact_mass"].to(device)

                token_logits, mass_pred = decoder(
                    input_ids=input_ids,
                    cond=cond,
                    token_mask=token_mask,
                )

                ce_loss = ce_loss_fn(
                    token_logits.reshape(-1, token_logits.size(-1)),
                    target_ids.reshape(-1),
                )

                mass_loss = ppm_quadratic_loss(
                    mass_pred=mass_pred,
                    mass_true=exact_mass,
                    alpha=1e-6,
                )

                loss = ce_loss + lambda_mass * mass_loss

                total_val_loss += loss.item()
                total_val_ce += ce_loss.item()
                total_val_mass += mass_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_ce = total_val_ce / len(val_loader)
        avg_val_mass = total_val_mass / len(val_loader)

        print(
            f"[Epoch {epoch + 1}/{num_epochs}] "
            f"train_total={avg_train_loss:.4f}, "
            f"train_ce={avg_train_ce:.4f}, "
            f"train_mass={avg_train_mass:.4f}, "
            f"val_total={avg_val_loss:.4f}, "
            f"val_ce={avg_val_ce:.4f}, "
            f"val_mass={avg_val_mass:.4f}"
        )