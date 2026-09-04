"""Measure real RSSM per-step cost before the model is built.

Plan 4's study is scoped from this number. M2's equivalent estimate predicted a
9.6x arm speed ratio where the measured value was 1.96x, so this exists to keep
the same mistake from setting a multi-hour compute budget.

The stand-in is shape- and structure-accurate: a GRUCell stepped `seq_len`
times at the real widths, plus heads at the real widths. Cost is determined by
tensor shapes and the sequential structure, not by trained weights.
"""

import argparse
import time

import torch
import torch.nn as nn

from mbfps.utils.device import get_device

H_DIM, Z_CATS, Z_CLASSES = 512, 32, 32
Z_DIM = Z_CATS * Z_CLASSES
LATENT_DIM = H_DIM + Z_DIM
EMBED_DIM = 2048
N_ACTIONS = 6


class _Standin(nn.Module):
    """Same shapes and same sequential structure as the real RSSM."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.GRUCell(Z_DIM + N_ACTIONS, H_DIM)
        self.prior = nn.Sequential(nn.Linear(H_DIM, 512), nn.SiLU(), nn.Linear(512, Z_DIM))
        self.post = nn.Sequential(
            nn.Linear(H_DIM + EMBED_DIM, 512), nn.SiLU(), nn.Linear(512, Z_DIM)
        )
        self.emb_head = nn.Sequential(
            nn.Linear(LATENT_DIM, 512), nn.SiLU(), nn.Linear(512, EMBED_DIM)
        )
        self.reward_head = nn.Sequential(
            nn.Linear(LATENT_DIM, 512), nn.SiLU(), nn.Linear(512, 1)
        )
        self.cont_head = nn.Sequential(
            nn.Linear(LATENT_DIM, 512), nn.SiLU(), nn.Linear(512, 1)
        )

    def forward(self, embeddings, actions):
        b, t, _ = embeddings.shape
        h = torch.zeros(b, H_DIM, device=embeddings.device)
        z = torch.zeros(b, Z_DIM, device=embeddings.device)
        outs = []
        for i in range(t):
            h = self.cell(torch.cat([z, actions[:, i]], dim=-1), h)
            _ = self.prior(h)
            logits = self.post(torch.cat([h, embeddings[:, i]], dim=-1))
            probs = torch.softmax(logits.view(b, Z_CATS, Z_CLASSES), dim=-1)
            z = probs.reshape(b, Z_DIM)
            outs.append(torch.cat([h, z], dim=-1))
        latents = torch.stack(outs, dim=1)
        return self.emb_head(latents), self.reward_head(latents), self.cont_head(latents)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    device = get_device(prefer=args.device)
    model = _Standin().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)

    embeddings = torch.randn(args.batch_size, args.seq_len, EMBED_DIM, device=device)
    actions = torch.zeros(args.batch_size, args.seq_len, N_ACTIONS, device=device)
    actions[:, :, 0] = 1.0

    for i in range(3):  # warm up: first steps pay kernel compilation
        loss = sum(o.square().mean() for o in model(embeddings, actions))
        optimiser.zero_grad(); loss.backward(); optimiser.step()
    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for i in range(args.steps):
        loss = sum(o.square().mean() for o in model(embeddings, actions))
        optimiser.zero_grad(); loss.backward(); optimiser.step()
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    ms = 1000 * elapsed / args.steps
    print(f"device={device} batch={args.batch_size} seq_len={args.seq_len}")
    print(f"ms_per_step={ms:.1f}")
    print(f"steps_per_second={1000 / ms:.3f}")
    print(f"hours_for_20k_steps={20_000 * ms / 3_600_000:.2f}")


if __name__ == "__main__":
    main()
