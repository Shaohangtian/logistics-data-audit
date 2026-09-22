# 数据说明

## `raw/` —— 本项目分析的主数据集（已入库）

| 项目 | 内容 |
|---|---|
| 文件 | `smart_logistics_dataset.csv` |
| 规模 | 1000 行 × 16 列 |
| 来源 | [Smart Logistics Supply Chain Dataset](https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset)（Kaggle，作者 ziya07） |
| 许可 | CC0: Public Domain |
| SHA256 | `FC20EA34C74FCEFE11F4FC8EDBFA353BA42151614E04008BB0607CC61BFE15A1` |

**本副本与 Kaggle 原文件不是逐字节相同的**，差异仅在序列化格式：

- 时间戳由 `2024-03-20 00:11:14` 变为 `2024/3/20 0:11`（精度截断到分钟）
- 浮点数去掉了尾部 `.0`（`27.0` → `27`）

**分析相关的字段完全一致**：分类字段与计数字段逐字段相同；
时间戳在重新格式化后仍保持 1000 个全部唯一、无重复。

> 校验方式：
> ```powershell
> Get-FileHash data/raw/smart_logistics_dataset.csv -Algorithm SHA256
> ```

### 来源元数据（便于核对）

以下信息取自 Kaggle 数据集 API（`/api/v1/datasets/view/ziya07/smart-logistics-supply-chain-dataset`），
用于证明上表的来源与许可声明是可查证的：

| 字段 | 值 |
|---|---|
| dataset id | 6614836 |
| 当前版本 | 1（2025-02-06 发布） |
| 许可 | **CC0: Public Domain** |
| 原始文件字节数 | 106,483 |
| 下载量 | 7,929 |

**许可结论**：CC0 属公共领域，可自由使用、修改、商用与再分发，无强制条件（注明来源是好习惯，但非义务）。

### 字段字典

> 下表是**实际读入本项目 CSV 后统计得到**的（非照抄 Kaggle 页面）。
> 范围按数据本身的小数位数给出，未做四舍五入。

| # | 列名 | 类型 | 含义 | 取值范围 / 类别 | 唯一值 |
|---|---|---|---|---|---|
| 0 | `Timestamp` | 字符串 | 数据记录时间 | 2024 全年 | 1000 |
| 1 | `Asset_ID` | 类别 | 运输资产编号 | `Truck_1` ~ `Truck_10` | 10 |
| 2 | `Latitude` | 浮点 | 纬度 | −89.79 ~ 89.87 | 1000 |
| 3 | `Longitude` | 浮点 | 经度 | −179.82 ~ 179.92 | 1000 |
| 4 | `Inventory_Level` | 整数 | 库存水平 | 100 ~ 500 | 366 |
| 5 | `Shipment_Status` | 类别 | 运输状态 | `Delivered` / `In Transit` / `Delayed` | 3 |
| 6 | `Temperature` | 浮点 | 温度（℃） | 18.0 ~ 30.0 | 121 |
| 7 | `Humidity` | 浮点 | 湿度（%） | 50.0 ~ 80.0 | 291 |
| 8 | `Traffic_Status` | 类别 | 路况 | `Clear` / `Detour` / `Heavy` | 3 |
| 9 | `Waiting_Time` | 整数 | 等待时长（分钟） | 10 ~ 60 | 51 |
| 10 | `User_Transaction_Amount` | 整数 | 用户交易金额 | 100 ~ 500 | 365 |
| 11 | `User_Purchase_Frequency` | 整数 | 用户购买频次 | 1 ~ 10 | 10 |
| 12 | `Logistics_Delay_Reason` | 类别 | 延迟原因 | `Weather` / `Traffic` / `Mechanical Failure` / `None` | 4 |
| 13 | `Asset_Utilization` | 浮点 | 资产利用率（%） | 60.0 ~ 100.0 | 366 |
| 14 | `Demand_Forecast` | 整数 | 需求预测 | 100 ~ 300 | 200 |
| 15 | **`Logistics_Delay`** | 0/1 | **目标变量：是否延迟** | `0` 准时 434 / `1` 延迟 566 | 2 |

### 读数据时务必注意

```python
df = pd.read_csv(path, keep_default_na=False, na_values=[''])
```

- 本表**没有任何真正的缺失值**（`isna().sum() == 0`）。
- 原因字段里的 `None` 是**有效取值**（263 行），不是缺失值。
  pandas 默认会把字符串 `None` 转成 `NaN`，从而抹掉这个取值，
  并让后续 `dropna()` 误删有效数据 —— 必须用上面的读法保住它。
- `Timestamp` 读进来是**字符串**，需要 `pd.to_datetime()` 才能按时间分组。

### 目标变量分布

| 取值 | 含义 | 行数 | 占比 |
|---|---|---|---|
| `1` | 延迟 | 566 | **56.6%** |
| `0` | 准时 | 434 | 43.4% |

多数类基线为 **56.6%** —— 任何分类模型必须显著超过这个数才算有效
（实测结果见 `notebooks/02_leakage_detection.ipynb` 末尾的实验）。

## `external/` —— 跨数据集验证用的第三方数据（**不入库**）

这些数据用于验证 `src/audit.py` 的通用性（手册 6.4）：
同一套检测逻辑，既要在合成数据上揪出泄漏，也要让真实数据通过。

| 文件 | 内容 | 来源 |
|---|---|---|
| `olist_orders_dataset.csv` | 巴西电商真实订单（99,441 行） | [olist/work-at-olist-data](https://github.com/olist/work-at-olist-data) |
| `olist_order_items_dataset.csv` | Olist 订单明细（112,650 行） | 同上 |
| `tips.csv` | 餐厅小费数据（244 行） | [mwaskom/seaborn-data](https://github.com/mwaskom/seaborn-data) |
| `titanic.csv` | 泰坦尼克乘客数据（891 行） | 同上 |

> **关于许可**：这四个数据集的许可**尚未核实**，使用时请自行到各自仓库页面确认。
>
> - Olist 数据集通常带有**非商用**条款，若用于商业场景请先确认。
> - `seaborn-data` 里的示例数据是从多个公开来源收集的，各文件许可可能不同。
>
> **对本项目无影响**：它们只用于本地验证 `src/audit.py` 的通用性，
> **不纳入版本控制、也不随本仓库分发**（见 `.gitignore`）。
> 而本项目的主数据集是 CC0，可自由使用。

### 下载方式

**体积约 31 MB，因此不纳入版本控制**（已写入 `.gitignore`）。
克隆仓库后用下面任一方式获取：

```powershell
# 方式一：一键下载 + 跑对比（推荐）
.venv\Scripts\python.exe scripts/compare_datasets.py --fetch

# 方式二：只跑对比（若已下载过）
.venv\Scripts\python.exe scripts/compare_datasets.py --brief
```

脚本会打印对比表，并把完整报告写到 `reports/cross_dataset_audit.md`。

### 关于 `titanic.csv`

它含有 `alive`、`class`、`who` 等**派生列**，其中 `alive` 就是 `survived` 的文字版，
因此会被目标泄漏检测命中——**这是真阳性，不是误报**。
所以它默认不参与对比（避免"真实数据也报严重泄漏"造成误解），
需要时用 `--include-titanic` 显式加入。
