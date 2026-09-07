# 弹性参数来源说明（elasticities.csv）

## 出处

当前取值来自 **GTAP 10 数据库 + Pothen & Hübler (2018) 结构估计**，
经 Braun (2025, Justus Liebig University Giessen 博士论文) 附录
Table B2 完整转载（原文截图核对过，非文本提取错列）：

> Model sectors (goods and services as defined by Pothen and Hübler (2018))
> and related elasticities of substitution. Armington elasticities are
> derived from the structural estimation of the Eaton and Kortum trade
> model by Pothen and Hübler (2018) according to the trade model
> unification theory of Arkolakis et al. (2012). The remaining trade and
> input elasticities are taken from the GTAP 10 database (Aguiar et al.,
> 2019).

GTAP 惯例：ESUBM（进口来源间替代）= 2 × ESUBD（国产-进口替代），
见 GTAP v6 文档（"rule of two"）。本表两列均直接取自文献，
未套用 rule of two。

## 原始表（Pothen & Hübler / GTAP 10，Braun 2025 Table B2）

| 部门 | σ_M（跨来源） | σ_DM（国产vs进口） |
|---|---|---|
| AGRI 农业 | 2.69 | 2.35 |
| COAL 煤 | 10.02 | 3.05 |
| CRUD 原油 | 7.89 | 5.20 |
| NGAS 天然气 | 7.94 | 12.96 |
| PETR 成品油 | 9.67 | 2.10 |
| FOOD 食品加工 | 3.80 | 2.48 |
| MINE 采矿 | 2.43 | 0.90 |
| PAPR 纸浆纸 | 5.18 | 2.95 |
| CHEM 化工橡塑 | 4.45 | 3.30 |
| NMMS 非金属矿物 | 6.39 | 2.90 |
| IRST 钢铁 | 4.21 | 2.95 |
| NFMS 有色金属 | 4.43 | 4.20 |
| MANU 制造业（综合） | 5.05 | 3.83 |
| ELEC 电力 | 18.66 | 2.80 |
| TRNS 运输 | 6.21 | 1.90 |
| CONS 建筑 | 15.07 | 1.90 |
| SERV 服务 | 6.43 | 1.92 |

## 映射到模型 12 部门（简单平均；多对一取 MANU）

| 模型部门 | 来源部门 | sigma (σ_M) | sigma_d (σ_DM) |
|---|---|---|---|
| AGF | AGRI + FOOD | 3.25 | 2.42 |
| MIN | MINE | 2.43 | 0.90 |
| ENR | COAL+CRUD+NGAS+PETR | 8.88 | 5.83 |
| CHM | CHEM | 4.45 | 3.30 |
| TXL | MANU | 5.05 | 3.83 |
| WDP | PAPR | 5.18 | 2.95 |
| NMM | NMMS+IRST+NFMS | 5.01 | 3.35 |
| MAC | MANU | 5.05 | 3.83 |
| ELE | MANU | 5.05 | 3.83 |
| VEH | MANU | 5.05 | 3.83 |
| OTM | MANU | 5.05 | 3.83 |
| SRV | SERV | 6.43 | 1.92 |

## 注意事项

- NGAS 的 σ_DM=12.96 > σ_M 是该结构估计的原值（非笔误）；ENR 部门取能源
  四类简单平均（5.83），受其影响偏高。
- MINE 的 σ_DM=0.90 < 1：CES 在 σ<1 时合法（互补品），数值上无奇异。
- 服务的 σ_M=6.43 来自 SERV 行；ELEC/CONS 的极高 σ_M（15-19）未纳入
  （我们的 SRV 含公用事业与建筑，但它们无贸易数据，不起作用）。
- 升级路径：若取得 GTAP 数据库许可，可直接用官方分部门 ESUBM/ESUBD
  替换本表（本表结构与之一一对应）。
