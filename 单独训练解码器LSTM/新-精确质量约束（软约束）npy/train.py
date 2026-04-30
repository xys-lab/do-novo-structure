# train.py
import csv
from pathlib import Path

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
    ppm = |mass_pred - mass_true| / mass_true * 1e6
    loss = alpha * ppm^2
    """
    ppm = torch.abs(mass_pred - mass_true) / torch.clamp(mass_true, min=1e-8) * 1e6
    loss = alpha * (ppm ** 2)
    return loss.mean()


def save_checkpoint(save_path, epoch, decoder, optimizer, best_val_loss):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        save_path,
    )


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
    num_workers=0,
    pin_memory=True,
    save_dir="checkpoints",
    use_amp=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda" and use_amp)

    print("Using device:", device)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("========== Entering training loop ==========")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Batch size:    {batch_size}")
    print(f"Epochs:        {num_epochs}")
    print("============================================")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    decoder.to(device)

    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=lr,
    )

    ce_loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id
    )

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    log_csv_path = save_dir / "train_log.csv"
    if not log_csv_path.exists():
        with open(log_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_total", "train_ce", "train_mass",
                "val_total", "val_ce", "val_mass",
                "lr",
            ])

    best_val_loss = float("inf")

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
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            cond = batch["cond_vec"].to(device, non_blocking=True)
            token_mask = batch["token_mask"].to(device, non_blocking=True)
            exact_mass = batch["exact_mass"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
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

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

        # =============================
        # Validation
        # =============================
        decoder.eval()
        total_val_loss = 0.0
        total_val_ce = 0.0
        total_val_mass = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                target_ids = batch["target_ids"].to(device, non_blocking=True)
                cond = batch["cond_vec"].to(device, non_blocking=True)
                token_mask = batch["token_mask"].to(device, non_blocking=True)
                exact_mass = batch["exact_mass"].to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, enabled=amp_enabled):
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

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch + 1}/{num_epochs}] "
            f"train_total={avg_train_loss:.4f}, "
            f"train_ce={avg_train_ce:.4f}, "
            f"train_mass={avg_train_mass:.4f}, "
            f"val_total={avg_val_loss:.4f}, "
            f"val_ce={avg_val_ce:.4f}, "
            f"val_mass={avg_val_mass:.4f}"
        )

        with open(log_csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                avg_train_loss, avg_train_ce, avg_train_mass,
                avg_val_loss, avg_val_ce, avg_val_mass,
                current_lr,
            ])

        # latest checkpoint
        save_checkpoint(
            save_path=save_dir / "latest.pt",
            epoch=epoch + 1,
            decoder=decoder,
            optimizer=optimizer,
            best_val_loss=best_val_loss,
        )

        # best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                save_path=save_dir / "best_val_total.pt",
                epoch=epoch + 1,
                decoder=decoder,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
            )
            print(f"[OK] 保存最优模型 -> {save_dir / 'best_val_total.pt'}")