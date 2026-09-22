# 来源与原创声明

## 参考的源项目
- 标题：Logistics Supply Chain Analysis - Lean Six Sigma
- 作者：Shiv（Kaggle shivambhardwaj23）
- 地址：https://www.kaggle.com/code/shivambhardwaj23/logistics-supply-chain-analysis-lean-six-sigma
- 数据集：https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset

## 本项目沿用的部分
- DMAIC（定义-测量-分析-改进-控制）分析框架
- 探索性数据分析（EDA）的整体流程与部分图表思路

## 本项目独立完成的部分
1. 数据质量审计工具 `src/audit.py`，实现五项检测：
   - 目标泄漏检测 `detect_target_leakage`（notebooks/02、src/audit.py）
   - 均匀随机性检验 `test_uniformity`（notebooks/03、src/audit.py）
   - 标签一致性检验 `check_label_consistency`（src/audit.py）
   - 极端比例检验 `check_extreme_ratio`（src/audit.py）
   - 类别不平衡 `class_balance`（src/audit.py）
   统一入口：`AuditReport` 报告类 + `audit()` 函数，支持"严重/警告/提示"三级标注。
2. 跨数据集通用性验证：用同一套 `audit()` 审计了本项目数据 + 3 份公开数据集
   （Olist 订单/订单明细、seaborn tips），结果见
   `scripts/compare_datasets.py` 与 `reports/cross_dataset_audit.md`。
   4 份中 1 份检出严重目标泄漏（本项目），3 份通过。
3. 目标泄漏的完整验证：目标变量可由 Shipment_Status 与 Traffic_Status
   完全复现，命中率 1000/1000
4. 延迟原因字段与"是否延迟"独立（卡方 p = 0.9925），四类原因分布近乎均匀
5. 时间维度分析（源项目完全未涉及）
6. 回归测试 `tests/test_audit.py`：30 个用例，含把上述数字锁死的真数据断言
7. 基线模型实验（notebooks/03 末尾，5 折分层交叉验证）：
   - A 用泄漏字段（2 个特征）→ 准确率 **1.0000**
   - B 删掉泄漏字段（7 个特征）→ **0.5290**
   - C 多数类基线 → **0.5660**
   即：删掉泄漏字段后不仅没有超过基线，反而低于基线 3.7 个百分点，
   证明数据中不存在可学习的信号。

## 许可
原始数据集许可：CC0: Public Domain
本仓库代码：MIT