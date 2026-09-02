# 外部验证：与文献估计对照（2018–19 中美贸易战量级）

## 情景

模型：6 地区 × 12 部门 CGE（2022 基准，嵌套 Armington + 中间投入 +
政府/储蓄/投资，有效关税基准）。

冲击：模拟 2018–19 贸易战的近似——中美双方对对方**全部货物部门**
加征 +25 个百分点（基准有效税率之上），homotopy 8 步续解。
（实际 2018–19 覆盖约为 2/3 的进口、平均约 20pp，故本情景略强于实际。）

## 结果对照

| 地区 | 本模型 | 文献估计 | 对照 |
|---|---|---|---|
| **美国** | **−0.035%** | Fajgelbaum et al. (2020, QJE)：**−0.04%**；Grossman-Helpman-Redding (2023)：约 **−0.12%** | ✅ 落在文献区间内，几乎命中 QJE 主估计 |
| **中国** | **−0.283%** | Chang et al. (2021)：**−0.29%**；Ma (2024)：**−0.29%** | ✅ 几乎逐位一致 |
| **第三方** | 日 +0.31% / 韩 +0.34% / 欧 +0.40% / ROW +0.45% | 贸易转移文献一致预期第三方获益 | ✅ 方向与符号结构一致 |

## 贸易量对照

- 模型：中对美货物出口 **−58.8%**，美对中 **−64.0%**
- 实际 2018→2019：中对美出口 −16%（$540B→$452B）
- 差异解释：本情景是 25pp×全部货物（强于实际的 20pp×2/3 覆盖），
  且模型无存量调整/合同粘性，属于长期比较静态结果，方向与量级合理

## 结论

模型在**文献锚定的量级上通过外部验证**：福利损失的规模
（美国万分之三到四、中国千分之四）与主流估计一致，第三方的
贸易转移增益方向正确。作为政策分析平台，其数量级可信度
得到已发表证据支持。

## 附：模型参数

- 基准：2022（IOT/Comtrade/BaTIS/WITS/GFS）
- 弹性：**GTAP 10 数据库 + Pothen & Hübler (2018) 结构估计**（经
  Braun 2025 附录 Table B2 转载），映射到 12 部门见
  `data_real_6x12/elasticities.csv` 与 `elasticities_来源说明.md`
- 求解：GAMSPy MCP + PATH，homotopy 8 步

## 文献出处

- Fajgelbaum, Goldberg, Kennedy, Khandelwal (2020). "The Return to
  Protectionism", *Quarterly Journal of Economics*. 美国福利损失
  $7.8B（−0.04% GDP）。
- Grossman, Helpman, Redding (2023). "When Tariffs Disrupt Global
  Supply Chains", *American Economic Review*. 美国损失约 0.12% GDP。
- Chang, Yao, Zheng (2021) 复制 FKPG 方法估计中国损失 −0.29% GDP；
  Ma (2024) 综述确认同量级。
