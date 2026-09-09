# bench_forward.py — 测量福利 NN 的前向传播(推理)时间 (CPU/GPU 自适应)
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = Path("outputs/welfare_nn")
ckpt = torch.load(BASE / "welfare_nn.pt", map_location="cpu", weights_only=False)
HIDDEN = ckpt["hidden"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "")


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


in_dim = ckpt["n_tariff"] + ckpt["n_years"]
model = WelfareNet(in_dim).to(DEVICE)
model.load_state_dict(ckpt["state"])
model.eval()
print(f"model: {in_dim} -> {'/'.join(map(str,HIDDEN))} -> 14")


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def bench(x, n_repeat):
    with torch.no_grad():
        model(x)  # warmup
        model(x)
        sync()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            model(x)
        sync()
        dt = time.perf_counter() - t0
    return dt / n_repeat


x1 = torch.randn(1, in_dim, device=DEVICE)
x10000 = torch.randn(10000, in_dim, device=DEVICE)
x100000 = torch.randn(100000, in_dim, device=DEVICE)

t1 = bench(x1, 1000)
t10k = bench(x10000, 100)
t100k = bench(x100000, 20)
print(f"\n=== 前向传播时间 ({DEVICE}) ===")
print(f"单场景 x 1:         {t1*1e6:8.1f} us/次")
print(f"批量 10000 场景:    {t10k*1e3:8.2f} ms/次  (每场景 {t10k/10000*1e6:.1f} us)")
print(f"批量 100000 场景:   {t100k*1e3:8.2f} ms/次  (每场景 {t100k/100000*1e6:.1f} us)")
print(f"\n对比 CGE 求解器:    单场景 ~410 ms (MC 实测)")
print(f"加速比 (单场景):    {410e3/t1/1e6:,.0f} x")
print(f"加速比 (批量10万):  {410e3/(t100k/100000)/1e6:,.0f} x")
