"""One synthetic optimization step with the GrassRoute optimizer."""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.grassroute import GrassRoute


def main():
    torch.manual_seed(42)
    layer = torch.nn.Linear(64, 32)
    optimizer = GrassRoute(
        [{"params": [layer.weight], "routed": True, "rank": 8},
         {"params": [layer.bias], "routed": False}],
        lr=1e-3,
        grassmann_lr=1e-3,
    )
    x = torch.randn(16, 64)
    y = torch.randn(16, 32)
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(layer(x), y)
    loss.backward()
    optimizer.step()
    print(f"one-step loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()
