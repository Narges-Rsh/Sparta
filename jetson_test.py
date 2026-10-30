import argparse
import os
import sys
import torch


# ============================================================
# SPARTA project path
# ============================================================

SPARTA_ROOT = "/home/nrashvan/sparta_test/Sparta"

if SPARTA_ROOT not in sys.path:
    sys.path.insert(0, SPARTA_ROOT)

from models import TransformerPredictor


def main():

    # ============================================================
    # Command-line arguments
    # ============================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained SPARTA checkpoint",
    )

    args = parser.parse_args()

    # ============================================================
    # Device
    # ============================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("========================================")
    print("DEVICE")
    print("========================================")
    print(device)

    # ============================================================
    # Load checkpoint FIRST
    # ============================================================

    print("\n========================================")
    print("LOADING CHECKPOINT")
    print("========================================")

    print(args.checkpoint)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    print(
        "Checkpoint keys:",
        checkpoint.keys(),
    )

    # ============================================================
    # Read original training arguments
    # ============================================================

    if "args" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain original training args."
        )

    ckpt_args = checkpoint["args"]

    print("\n========================================")
    print("ORIGINAL TRAINING ARGUMENTS")
    print("========================================")

    if hasattr(ckpt_args, "__dict__"):

        for key, value in vars(ckpt_args).items():
            print(f"{key}: {value}")

    elif isinstance(ckpt_args, dict):

        for key, value in ckpt_args.items():
            print(f"{key}: {value}")

    # ============================================================
    # Helper for reading saved arguments
    # ============================================================

    def get_arg(name, default=None):

        if isinstance(ckpt_args, dict):
            return ckpt_args.get(name, default)

        return getattr(
            ckpt_args,
            name,
            default,
        )

    # ============================================================
    # Get ORIGINAL configuration
    # ============================================================

    seg_len = get_arg(
        "seg_len",
        24,
    )

    num_kp = get_arg(
        "num_kp",
        18,
    )

    token_config = get_arg(
        "token_config",
        "CTR",
    )

    relative = get_arg(
        "relative",
        False,
    )

    traj = get_arg(
        "traj",
        False,
    )

    num_heads = get_arg(
        "model_num_heads",
        get_arg("num_heads", 2),
    )

    latent_dim = get_arg(
        "model_latent_dim",
        get_arg("latent_dim", 64),
    )

    num_layers = get_arg(
        "model_num_layers",
        get_arg("num_layers", 2),
    )

    dropout = get_arg(
        "dropout",
        0.1,
    )

    # ============================================================
    # Reproduce original d_model logic
    # ============================================================

    expand_ratio = 1
    extra_dim = 0

    if relative:
        expand_ratio = 2

    if token_config == "kps":

        input_dim = seg_len * 2

    elif token_config == "2ds":

        input_dim = seg_len

    elif token_config == "st":

        input_dim = 2

    else:

        if traj:

            if relative:
                extra_dim = 4

            else:
                extra_dim = 2

        input_dim = num_kp * 2

    d_model = (
        input_dim * expand_ratio
        + extra_dim
    )

    # ============================================================
    # Print reconstructed architecture
    # ============================================================

    print("\n========================================")
    print("MODEL CONFIGURATION")
    print("========================================")

    print("token_config :", token_config)
    print("seg_len      :", seg_len)
    print("num_kp       :", num_kp)
    print("relative     :", relative)
    print("traj         :", traj)

    print("d_model      :", d_model)
    print("num_heads    :", num_heads)
    print("latent_dim   :", latent_dim)
    print("num_layers   :", num_layers)
    print("dropout      :", dropout)

    # ============================================================
    # Create model
    # ============================================================

    model = TransformerPredictor(
        d_model,
        num_heads,
        latent_dim,
        num_layers,
        1000,
        device=str(device),
        dropout=dropout,
    )

    model = model.to(device)

    # ============================================================
    # Load pretrained weights
    # ============================================================

    state_dict = checkpoint["state_dict"]

    clean_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        clean_state_dict[key] = value

    model.load_state_dict(
        clean_state_dict,
        strict=True,
    )

    model.eval()

    print(
        "\nPretrained SPARTA model loaded successfully."
    )

    # ============================================================
    # Parameter count
    # ============================================================

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\n========================================")
    print("MODEL INFORMATION")
    print("========================================")

    print(
        f"Total parameters     : {total_params:,}"
    )

    print(
        f"Trainable parameters : {trainable_params:,}"
    )

    print(
        f"Total parameters (K) : {total_params / 1e3:.2f} K"
    )

    print(
        f"Total parameters (M) : {total_params / 1e6:.4f} M"
    )

    # ============================================================
    # Random Transformer input
    #
    # Transformer expects:
    #
    # [Batch, Sequence, d_model]
    #
    # Your checkpoint indicates d_model = 72.
    # ============================================================

    x = torch.randn(
        1,
        seg_len,
        d_model,
        dtype=torch.float32,
        device=device,
    )

    print("\n========================================")
    print("INPUT")
    print("========================================")

    print(
        "Input shape:",
        x.shape,
    )

    print(
        "Input dtype:",
        x.dtype,
    )

    print(
        "Input device:",
        x.device,
    )

    print(
        "Input min:",
        x.min().item(),
    )

    print(
        "Input max:",
        x.max().item(),
    )

    # ============================================================
    # Inference
    # ============================================================

    print("\n========================================")
    print("INFERENCE")
    print("========================================")

    with torch.no_grad():

        recon = model(
            x,
            x,
        )

    # ============================================================
    # Output
    # ============================================================

    print("\n========================================")
    print("OUTPUT")
    print("========================================")

    print(
        "Reconstruction shape:",
        recon.shape,
    )

    print(
        "Reconstruction min:",
        recon.min().item(),
    )

    print(
        "Reconstruction max:",
        recon.max().item(),
    )

    print(
        "Reconstruction mean:",
        recon.mean().item(),
    )

    print(
        "Reconstruction std:",
        recon.std().item(),
    )

    # ============================================================
    # Reconstruction error
    # ============================================================

    mse = torch.mean(
        (x - recon) ** 2
    )

    print("\n========================================")
    print("RECONSTRUCTION ERROR")
    print("========================================")

    print(
        "MSE:",
        mse.item(),
    )

    print("\n========================================")
    print("TEST COMPLETED SUCCESSFULLY")
    print("========================================")


if __name__ == "__main__":
    main()