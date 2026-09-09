# train_welfare_nn.py — 全连接神经网络拟合 MC 福利响应
#
# 输入: 反事实关税 (MFN 口径 168 维 = 14 目的国 x 12 部门) + 年份 one-hot(15)
# 输出: 14 国 welfare_pct
# 数据: 15 年 x 10000 已解场景 = 150000 样本
#       (taus.npy 只取前 10000 行; 关税向量为所有来源相同的 MFN 税率)
from __future__ import annotations
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = Path(os.environ.get("MC_BASE", "outputs/mc_exports_2005_2019"))
YEARS = list(range(2005, 2020))
N_SOLVED = 10000
R, I = 14, 12           # 目的国 x 部门 = 168 个 MFN 关税
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 10000
BATCH = 512
LR = 1e-3
HIDDEN = (512, 512, 256, 128)
VAL, TEST = 0.1, 0.1


def load_dataset(cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (n,168), Y (n,14), year_idx (n,)."""
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["Y"], z["year_idx"]
    Xs, Ys, YI = [], [], []
    for y, yy in enumerate(YEARS):
        d = np.load(BASE / f"mc_{yy}" / "results.npz")
        w = d["welfare_pct"][:N_SOLVED]                       # (10000, 14)
        # MFN 税率: 对目的国 s 取来源 (s+1)%R (任一非自身来源, 各来源相同;
        # 不能用固定来源0, 否则目的国=来源0 时取到恒为 0 的国内对角线)
        t = np.load(BASE / f"mc_{yy}" / "taus.npy", mmap_mode="r")
        tau_mfn = np.empty((N_SOLVED, R, I), np.float32)
        for s in range(R):
            tau_mfn[:, s, :] = t[:N_SOLVED, (s + 1) % R, :, s]
        tau_mfn = tau_mfn.reshape(N_SOLVED, -1)
        Xs.append(tau_mfn); Ys.append(w); YI.append(np.full(N_SOLVED, y))
        print(f"  loaded {yy}: X {tau_mfn.shape}", flush=True)
    X = np.concatenate(Xs); Y = np.concatenate(Ys).astype(np.float32)
    year_idx = np.concatenate(YI)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, Y=Y, year_idx=year_idx)
    return X, Y, year_idx


class WelfareNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 14):
        super().__init__()
        layers, d = [], in_dim
        for h in HIDDEN:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main() -> None:
    torch.manual_seed(SEED); np.random.seed(SEED)
    t0 = time.perf_counter()
    cache = Path("nn_cache/dataset.npz")
    X, Y, year_idx = load_dataset(cache)
    print(f"dataset: X {X.shape}, Y {Y.shape} ({time.perf_counter()-t0:.1f}s)",
          flush=True)

    # 标准化输入; 输出保留原尺度(百分比), 但按 std 缩放以平衡各国量级
    x_mean, x_std = X.mean(0), X.std(0) + 1e-8
    y_std = Y.std(0) + 1e-8
    Xn = (X - x_mean) / x_std
    Yn = Y / y_std

    # 年份 one-hot 拼接到输入
    oh = np.eye(len(YEARS), dtype=np.float32)[year_idx]
    Xn = np.concatenate([Xn, oh], axis=1)

    n = len(Xn)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_test, n_val = int(n * TEST), int(n * VAL)
    idx = dict(test=perm[:n_test], val=perm[n_test:n_test + n_val],
               train=perm[n_test + n_val:])

    def tens(a):
        return torch.tensor(a, dtype=torch.float32, device=DEVICE)

    Xs = {k: tens(Xn[v]) for k, v in idx.items()}
    Ys = {k: tens(Yn[v]) for k, v in idx.items()}
    print(f"train {len(idx['train'])} val {len(idx['val'])} test {len(idx['test'])}"
          f"  device={DEVICE}", flush=True)

    model = WelfareNet(Xn.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    best_val, best_state, patience = float("inf"), None, 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_t = time.perf_counter()
        order = rng.permutation(len(idx["train"]))
        tot = 0.0
        for b in range(0, len(order), BATCH):
            bs = order[b:b + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xs["train"][bs]), Ys["train"][bs])
            loss.backward(); opt.step()
            tot += loss.item() * len(bs)
        model.eval()
        with torch.no_grad():
            v = lossf(model(Xs["val"]), Ys["val"]).item()
        if v < best_val - 1e-7:
            best_val, best_state, patience = v, \
                {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            patience += 1
        if ep % 10 == 0:
            print(f"epoch {ep:3d} train {tot/len(order):.3e} val {v:.3e}"
                  f" ({time.perf_counter()-ep_t:.1f}s)", flush=True)
        if patience >= 50:
            print(f"early stop at epoch {ep}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xs["test"]).cpu().numpy() * y_std
        true = Ys["test"].cpu().numpy() * y_std
    regs = list(np.load(BASE / "mc_2005" / "results.npz")["regions"])
    rmse = np.sqrt(((pred - true) ** 2).mean(0))
    r2 = 1 - ((pred - true) ** 2).sum(0) / ((true - true.mean(0)) ** 2).sum(0)
    print("\n=== test set (per country) ===")
    for i, r in enumerate(regs):
        print(f"{r:4s} RMSE {rmse[i]:.4f}  R2 {r2[i]:.5f}")
    print(f"overall RMSE {np.sqrt(((pred-true)**2).mean()):.4f}"
          f"  mean R2 {r2.mean():.5f}")

    out = Path(os.environ.get("NN_OUTDIR", "outputs/welfare_nn"))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=best_state, x_mean=x_mean, x_std=x_std, y_std=y_std,
                    n_tariff=R * I, n_years=len(YEARS), hidden=HIDDEN),
               out / "welfare_nn.pt")
    json.dump(dict(rmse=dict(zip(regs, rmse.tolist())),
                   r2=dict(zip(regs, r2.tolist())),
                   overall_rmse=float(np.sqrt(((pred - true) ** 2).mean())),
                   mean_r2=float(r2.mean()),
                   n_samples=int(n), device=str(DEVICE),
                   wall_seconds=time.perf_counter() - t0),
              open(out / "metrics.json", "w"), indent=2)
    print(f"\nsaved -> {out}  (total {time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
