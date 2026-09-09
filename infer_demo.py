# infer_demo.py — 用训练好的福利 NN 做推理并展示 预测 vs 真值
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

ckpt = torch.load("outputs/welfare_nn/welfare_nn.pt", map_location="cpu",
                  weights_only=False)
HIDDEN = ckpt["hidden"]
in_dim = ckpt["n_tariff"] + ckpt["n_years"]


class WelfareNet(nn.Module):
    def __init__(self, in_dim, out_dim=14):
        super().__init__()
        layers, d = [], in_dim
        for h in HIDDEN:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = WelfareNet(in_dim)
model.load_state_dict(ckpt["state"])
model.eval()

z = np.load("nn_cache/dataset.npz")
X, Y, yi = z["X"], z["Y"], z["year_idx"]
regs = list(np.load("outputs/mc_exports_2005_2019/mc_2005/results.npz")["regions"])

Xn = (X - ckpt["x_mean"]) / ckpt["x_std"]
oh = np.eye(15, dtype=np.float32)[yi]
Xn = np.concatenate([Xn, oh], axis=1)

with torch.no_grad():
    pred = model(torch.tensor(Xn, dtype=torch.float32)).numpy() * ckpt["y_std"]

# 与训练时相同的划分 (seed=42, val/test 各 10%)
rng = np.random.default_rng(42)
perm = rng.permutation(len(Xn))
n_test = int(len(Xn) * 0.1)
test_idx = perm[:n_test]

# ---- 1) 三个测试场景的 预测 vs 真值 ----
for si in test_idx[:3]:
    yr = 2005 + yi[si]
    print(f"\n=== 样本#{si} ({yr}年场景) 预测 vs 真值 (welfare_pct) ===")
    print(f"{'国家':5s} {'真值':>9s} {'预测':>9s} {'误差':>9s}")
    for i, r in enumerate(regs):
        t, p = Y[si, i], pred[si, i]
        print(f"{r:5s} {t:9.4f} {p:9.4f} {p-t:+9.4f}")

# ---- 2) 测试集整体误差分布 ----
err = pred[test_idx] - Y[test_idx]
print("\n=== 测试集误差分布 (15000 场景 x 14 国 = 210000 个预测点) ===")
print(f"mean  {err.mean():+.5f}   std {err.std():.5f}")
print(f"|误差| p50 {np.percentile(np.abs(err),50):.4f}"
      f"  p95 {np.percentile(np.abs(err),95):.4f}"
      f"  p99 {np.percentile(np.abs(err),99):.4f}"
      f"  max {np.abs(err).max():.4f}")
