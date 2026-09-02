# data_real_6x12：6 地区 × 12 部门基准数据集（2022）

由 `prepare_real_data_6x12.py` 生成；关税由 `prepare_baseline_tariffs_6x12.py` +
`prepare_effective_tariffs.py` 生成。服务流由 `prepare_batis_services.py` 生成。

## 文件

| 文件 | 内容 |
|---|---|
| `production_2022.csv` | 地区×部门：总产出 Y0、劳动报酬、资本（残差） |
| `bilateral_flows_2022.csv` | 来源×部门×目的地双边流量（百万美元，含国内流） |
| `io_coefficients_2022.csv` | 中间投入系数（投入部门×地区×产出部门），含 φ 调和 |
| `final_demand_2022.csv` | 最终需求三分：C0（家庭）、G0（政府）、I0（投资） |
| `net_transfers_2022.csv` | 净国外转移 B = M − X |
| `baseline_tariffs_2022.csv` | 名义基准关税（MFN/指标口径，未修正） |
| `tariff_effective_factors.csv` | 有效关税修正因子（G1151 锚定）：CHN 0.350 / USA 2.618 / JPN 0.624 / KOR 0.180 / EU27 0.412（EC 口径：2022 年上缴 250 亿欧元 ÷ 0.75 留存折算 × 汇率 1.053）/ ROW 0.261（37/49 经济体 GFS 覆盖，S13→S1311→S1311B 回退链） |
| `baseline_tariffs_effective_2022.csv` | 修正后有效关税（模型实际使用） |
| `services_flows_batis_2022.csv` / `services_shares_batis_2022.csv` | BaTIS 服务贸易矩阵与份额 |
| `elasticities.csv` | Armington 弹性：**GTAP 10 + Pothen & Hübler (2018)**（出处与映射见 `elasticities_来源说明.md`） |
| `sector_mapping.csv` | SITC 2 位 ↔ ISIC Rev.4 ↔ 模型部门对照及判断依据 |

## 已知妥协（按影响排序）

1. **服务贸易**：份额用 BaTIS 实际双边值；中美日韩的控制总量用 IOT，EU27 对外服务贸易总量用 BaTIS 实际值。SRV 未细分。
2. **关税**：结构为 MFN/指标口径 + G1151 有效修正（CHN/USA/JPN/KOR 用 IMF GFS，EU27 用欧委会传统自有财源口径）。ROW 未修正（GFS 逐国下载可行，待做）；从量税未换算 AVE（农业保护被低估）；美国 301 关税仅通过 2.618 总量因子体现，未分部门。
3. **SITC 9x 剔除**（2022 年约 1.55 万亿美元：特殊交易 93、黄金 97、硬币 96）——无生产账户对应；对应产出留在国内吸收残差。详见 `sector_mapping.csv` 末行。
4. **SITC↔ISIC 映射**：废金属（28）归 NMM、仪器（87/88）归 ELE 等判断性调整，依据见映射表 note 列。根治方案：重新以 HS6 下载 Comtrade（HS→ISIC 映射干净），工作量大，暂缓。
5. **io 调和**：φ ∈ [0.61, 1.46]，吸收 CIF/FOB 差与映射噪声。
6. **流量地板**：117/432 个流量单元地板至 100 万美元（CES 正性要求），造成每地区约 −1,200 万美元的闭合残差（模型断言以 atol 覆盖）。
7. **EU27**：IOT 内部含欧盟内部贸易，用 Comtrade 对外比例修正货物（0.597/0.666），服务用 BaTIS 实际对外值。
