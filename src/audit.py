# -*- coding: utf-8 -*-
"""
数据质量审计工具（data quality audit）
====================================

针对"这份数据能不能用来建模"这一最基础的问题，提供五项可复用的检测。

设计目标：**对任何数据集都能工作**——所有阈值都是参数，区间可以自动推断，
不针对某一份数据硬编码。

五项检测
--------
1. detect_target_leakage  目标泄漏：某字段（或字段组合）能否完全决定目标变量
2. test_uniformity        均匀随机性：数值字段是否像随机生成而非真实观测
3. check_label_consistency 标签一致性：标志位与原因字段是否自相矛盾
4. check_extreme_ratio    极端比例：是否存在 0% / 100% 这种确定性分组
5. class_balance          类别不平衡：多数类基线是多少

统一入口
--------
    from src.audit import audit
    report = audit(df, target='Logistics_Delay')
    report.print_summary()

作者说明：本模块由本项目的分析结论提炼而成，参见 ATTRIBUTION.md。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    'AuditReport',
    'audit',
    'detect_target_leakage',
    'test_uniformity',
    'check_label_consistency',
    'check_extreme_ratio',
    'class_balance',
    'default_numeric_ranges',
    'SEVERITY_ORDER',
]

# ---------------------------------------------------------------------------
# 通用常量
# ---------------------------------------------------------------------------

#: 三级严重程度。排序时用 SEVERITY_ORDER，'严重' 排在最前。
SEVERITY_ORDER = {'严重': 0, '警告': 1, '提示': 2}

#: 均匀性判定的默认容差：实际标准差 / 理论标准差不超出 1 ± tol 即视为均匀。
UNIFORM_TOL = 0.10

#: 视为"未填写"的哨兵字符串。CSV 里常见把缺失写成这些文本。
DEFAULT_SENTINELS = ('None', 'none', 'NONE', 'null', 'NULL', 'nan', 'NaN',
                     'na', 'NA', 'N/A', '-', '')


# ===========================================================================
# 检测一：目标泄漏
# ===========================================================================

def detect_target_leakage(df: pd.DataFrame,
                          target: str,
                          max_cardinality: int = 10,
                          max_combo_size: int = 2,
                          exclude: Iterable[str] = ()) -> list[dict]:
    """找出能单独或组合完全决定目标变量的字段。

    原理
    ----
    把所有行按候选字段分组；若**每一组**内部目标变量的唯一取值个数都是 1，
    说明该字段（组合）与目标变量存在函数关系——目标变量已被写死在特征里。
    这就是**目标泄漏**，它会让任何模型的准确率虚高到接近 100%。

    Parameters
    ----------
    df : DataFrame
    target : str
        目标变量列名。
    max_cardinality : int
        只把取值数不超过该值的字段当作候选（分类字段）。
    max_combo_size : int
        组合的最大字段数，默认 2（两两组合）。
    exclude : Iterable[str]
        显式排除的字段名，例如已经知道属于"结果"的列。

    Returns
    -------
    list[dict]
        每个元素形如 ``{'columns': [...], 'n_groups': int, 'accuracy': 1.0}``。
        空列表表示未检出泄漏。
    """
    if target not in df.columns:
        raise KeyError(f'目标变量 {target!r} 不在数据中')

    excluded = {target, *exclude}
    candidates = [c for c in df.columns
                  if c not in excluded
                  and df[c].nunique(dropna=True) <= max_cardinality]

    findings: list[dict] = []
    for size in range(1, max_combo_size + 1):
        for combo in combinations(candidates, size):
            keys = list(combo)
            nun = df.groupby(keys, observed=True, dropna=False)[target].nunique()
            if len(nun) and (nun == 1).all():
                findings.append({
                    'columns': keys,
                    'n_groups': int(len(nun)),
                    'accuracy': 1.0,
                })
    # 字段少的组合优先（更可能是"根因"）
    findings.sort(key=lambda f: (len(f['columns']), f['columns']))
    return findings


# ===========================================================================
# 检测二：均匀随机性
# ===========================================================================

def test_uniformity(series: pd.Series,
                    low: float | None = None,
                    high: float | None = None,
                    tol: float = UNIFORM_TOL) -> dict:
    """比较实际标准差与均匀分布的理论标准差。

    原理
    ----
    若 X ~ U(low, high)，则 ``sd(X) = (high - low) / sqrt(12)``。
    真实业务数据几乎不会精确落在这个理论值上；一旦落入，
    它极可能是用 ``random.uniform(low, high)`` 生成的，而非观测所得。

    Parameters
    ----------
    series : Series
        数值列。
    low, high : float, optional
        取值区间的上下限。默认用该列的 min / max 代替。
    tol : float
        容差，比值落在 ``1 ± tol`` 内视为均匀。

    Returns
    -------
    dict
        ``actual_sd`` / ``theoretical_sd`` / ``ratio`` / ``is_uniform``
        以及实际使用的 ``low`` / ``high`` 与缺失值个数。
    """
    s = pd.to_numeric(series, errors='coerce').dropna()
    n = int(len(s))
    if n < 2:
        return {'n': n, 'actual_sd': None, 'theoretical_sd': None,
                'ratio': None, 'is_uniform': False, 'low': low, 'high': high,
                'note': '有效样本不足，无法检验'}

    lo = float(s.min()) if low is None else float(low)
    hi = float(s.max()) if high is None else float(high)

    actual = float(s.std(ddof=1))
    theoretical = (hi - lo) / np.sqrt(12)
    if theoretical == 0:
        return {'n': n, 'actual_sd': actual, 'theoretical_sd': 0.0,
                'ratio': None, 'is_uniform': False, 'low': lo, 'high': hi,
                'note': '区间宽度为 0，无法检验'}

    ratio = actual / theoretical
    return {
        'n': n,
        'low': lo,
        'high': hi,
        'actual_sd': round(actual, 4),
        'theoretical_sd': round(theoretical, 4),
        'ratio': round(ratio, 4),
        'is_uniform': bool(abs(ratio - 1.0) <= tol),
    }


# 本函数的名字以 "test_" 开头（沿用手册的命名），但它是给用户调用的
# 审计函数，不是 pytest 测试用例。打上这个标记，避免被 pytest 误收集。
test_uniformity.__test__ = False


def default_numeric_ranges(df: pd.DataFrame,
                           exclude: Sequence[str] = (),
                           min_n: int = 20) -> dict[str, tuple[float, float]]:
    """为所有数值列自动推断 ``(min, max)`` 区间，供 test_uniformity 使用。

    这样审计工具不必为每份数据手写区间表。
    """
    out: dict[str, tuple[float, float]] = {}
    for col in df.columns:
        if col in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(s) >= min_n and s.nunique() > 2:
            out[col] = (float(s.min()), float(s.max()))
    return out


# ===========================================================================
# 检测三：标签一致性
# ===========================================================================

def check_label_consistency(df: pd.DataFrame,
                            flag_col: str,
                            reason_col: str,
                            positive: Any = 1,
                            sentinels: Sequence[str] = DEFAULT_SENTINELS
                            ) -> dict:
    """检查"标志位"与"原因"字段之间是否自相矛盾。

    原理
    ----
    业务上这两个字段通常互相蕴含：标志位为"是"就该有原因，为"否"就不该有。
    矛盾组合的比例越高，说明该字段不是真实记录，而是随机贴的标签。

    注意
    ----
    字符串 ``'None'`` 默认被当作**无原因**（而不是缺失值）。这一点很关键：
    `pandas.read_csv` 默认会把 ``None`` 转成 NaN，从而抹掉一个有效取值。
    读数据时请用 ``keep_default_na=False, na_values=['']``。

    Returns
    -------
    dict
        ``count`` / ``rate`` / ``detail`` 三类矛盾的计数。
    """
    if flag_col not in df.columns or reason_col not in df.columns:
        raise KeyError(f'{flag_col!r} 或 {reason_col!r} 不在数据中')

    reason = df[reason_col]
    sent = set(sentinels)
    # "有原因" = 非空 且 不是哨兵文本
    has_reason = reason.notna() & ~reason.astype(str).isin(sent)
    flagged = df[flag_col] == positive

    a = int((~flagged & has_reason).sum())    # 没延迟却写了原因
    b = int((flagged & ~has_reason).sum())    # 延迟了却没写原因
    total = len(df)
    return {
        'n_rows': total,
        'count': a + b,
        'rate': round((a + b) / total, 4) if total else 0.0,
        'detail': {'flag_false_but_has_reason': a,
                   'flag_true_but_no_reason': b},
    }


# ===========================================================================
# 检测四：极端比例
# ===========================================================================

def check_extreme_ratio(df: pd.DataFrame,
                        group_col: str,
                        target: str,
                        positive: Any = 1) -> dict[str, dict]:
    """找出目标比例恰好为 0% 或 100% 的分组。

    原理
    ----
    真实数据总会有反例：路况再差也可能准时，路况再好也可能延误。
    某个分组"一个反例都没有"，通常意味着数据是用规则拼装出来的。

    Returns
    -------
    dict
        形如 ``{'Heavy': {'n': 327, 'n_positive': 327, 'rate': 1.0}}``。
        空字典表示没有极端分组。
    """
    if group_col not in df.columns or target not in df.columns:
        raise KeyError(f'{group_col!r} 或 {target!r} 不在数据中')

    sub = df[[group_col, target]].copy()
    sub['_y'] = (sub[target] == positive).astype(int)
    g = sub.groupby(group_col, observed=True)['_y'].agg(['size', 'sum', 'mean'])

    out: dict[str, dict] = {}
    for key, row in g.iterrows():
        if row['mean'] in (0.0, 1.0):
            out[str(key)] = {'n': int(row['size']),
                             'n_positive': int(row['sum']),
                             'rate': float(row['mean'])}
    return out


# ===========================================================================
# 检测五：类别不平衡
# ===========================================================================

def class_balance(df: pd.DataFrame, target: str) -> dict:
    """统计目标变量的类别分布与多数类基线。

    原理
    ----
    任何分类模型都必须打败"永远猜多数类"这个基线。
    不报告基线的准确率是没有意义的。
    """
    if target not in df.columns:
        raise KeyError(f'目标变量 {target!r} 不在数据中')

    vc = df[target].value_counts(normalize=True, dropna=False)
    return {
        'distribution': {str(k): round(float(v), 4) for k, v in vc.items()},
        'majority_class': str(vc.idxmax()),
        'majority_baseline': round(float(vc.max()), 4),
        'n_classes': int(df[target].nunique(dropna=False)),
    }


# ===========================================================================
# 统一入口
# ===========================================================================

@dataclass
class AuditReport:
    """一次审计的结果。按"严重 / 警告 / 提示"三级排序，便于按优先级阅读。"""

    dataset_name: str
    n_rows: int
    n_cols: int
    findings: list[dict] = field(default_factory=list)

    # -- 写入 ---------------------------------------------------------------
    def add(self, severity: str, title: str, detail: str) -> None:
        if severity not in SEVERITY_ORDER:
            raise ValueError(f'severity 必须是 {list(SEVERITY_ORDER)} 之一')
        self.findings.append({'severity': severity,
                              'title': title,
                              'detail': detail})

    # -- 读取 ---------------------------------------------------------------
    @property
    def sorted_findings(self) -> list[dict]:
        return sorted(self.findings, key=lambda f: SEVERITY_ORDER[f['severity']])

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f['severity'] == severity)

    @property
    def has_critical(self) -> bool:
        return self.count('严重') > 0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.findings or [{'severity': '', 'title': '',
                                               'detail': ''}])

    # -- 输出 ---------------------------------------------------------------
    def print_summary(self) -> None:
        print(f'数据集：{self.dataset_name}  ({self.n_rows} 行 × {self.n_cols} 列)')
        if not self.findings:
            print('  未发现问题。')
            return
        for f in self.sorted_findings:
            print(f"[{f['severity']}] {f['title']}")
            print(f"    {f['detail']}")
        print()
        print(f"合计：严重 {self.count('严重')} · "
              f"警告 {self.count('警告')} · "
              f"提示 {self.count('提示')}")

    def to_markdown(self) -> str:
        lines = [f"# 数据质量审计报告：{self.dataset_name}", '',
                 f'- 规模：{self.n_rows} 行 × {self.n_cols} 列',
                 f"- 结论：严重 {self.count('严重')} 项，"
                 f"警告 {self.count('警告')} 项，提示 {self.count('提示')} 项", '']
        for f in self.sorted_findings:
            lines.append(f"## [{f['severity']}] {f['title']}")
            lines.append('')
            lines.append(f['detail'])
            lines.append('')
        return '\n'.join(lines)


def audit(df: pd.DataFrame,
          target: str,
          dataset_name: str = '未命名',
          numeric_ranges: dict[str, tuple[float, float]] | None = None,
          flag_col: str | None = None,
          reason_col: str | None = None,
          max_cardinality: int = 10,
          uniform_tol: float = UNIFORM_TOL,
          verbose: bool = False) -> AuditReport:
    """对一份带目标变量的数据集做五项数据质量检测，返回 AuditReport。

    Parameters
    ----------
    df : DataFrame
    target : str
        目标变量列名。
    dataset_name : str
        报告里显示的数据集名称。
    numeric_ranges : dict, optional
        数值列的 ``{列名: (low, high)}``。默认自动推断。
    flag_col, reason_col : str, optional
        若提供，则额外做"标签一致性"检测。
    max_cardinality : int
        目标泄漏检测时视为分类字段的最大取值数。
    uniform_tol : float
        均匀性容差。
    verbose : bool
        是否边跑边打印进度。

    Returns
    -------
    AuditReport
    """
    r = AuditReport(dataset_name, len(df), df.shape[1])

    leak_exclude = [c for c in (flag_col, reason_col) if c]

    # --- 1. 目标泄漏 ------------------------------------------------------
    hits = detect_target_leakage(df, target,
                                 max_cardinality=max_cardinality,
                                 exclude=leak_exclude)
    if hits:
        for h in hits[:5]:
            cols = ' + '.join(h['columns'])
            r.add('严重', '检测到目标泄漏',
                  f"字段 [{cols}] 可 100% 复现 {target}"
                  f"（{h['n_groups']} 个分组，每组目标值唯一）。"
                  f"该数据集不能用于建模：模型会学到恒等关系而非真实规律。")
        if len(hits) > 5:
            r.add('严重', '检测到目标泄漏（续）',
                  f'另有 {len(hits) - 5} 组字段组合同样可完全复现 {target}。')
    else:
        r.add('提示', '未检出目标泄漏',
              f'没有任何低基数字段或两两组合能完全决定 {target}。')
    if verbose:
        print(f'[1/5] 目标泄漏：{len(hits)} 处')

    # --- 2. 均匀随机性 ----------------------------------------------------
    ranges = numeric_ranges if numeric_ranges is not None \
        else default_numeric_ranges(df, exclude=[target])
    uni_rows = []
    for col, (lo, hi) in ranges.items():
        res = test_uniformity(df[col], lo, hi, tol=uniform_tol)
        res['field'] = col
        uni_rows.append(res)
    uniform_fields = [x['field'] for x in uni_rows if x['is_uniform']]
    if uni_rows:
        if len(uniform_fields) == len(uni_rows):
            r.add('警告', '全部数值字段都像均匀随机数',
                  f'{len(uniform_fields)}/{len(uni_rows)} 个数值字段的实际标准差'
                  f'与均匀分布理论值偏差在 ±{uniform_tol:.0%} 内：'
                  f'{", ".join(uniform_fields)}。'
                  f'真实观测数据几乎不会如此"标准"，提示数据可能是合成的。')
        elif uniform_fields:
            r.add('提示', '部分数值字段像均匀随机数',
                  f'{len(uniform_fields)}/{len(uni_rows)} 个字段通过均匀性检验：'
                  f'{", ".join(uniform_fields)}。')
        else:
            r.add('提示', '数值字段不符合均匀分布',
                  f'0/{len(uni_rows)} 个字段通过均匀性检验，'
                  f'这更像真实观测数据。')
    if verbose:
        print(f'[2/5] 均匀随机性：{len(uniform_fields)}/{len(uni_rows)} 通过')

    # --- 3. 标签一致性 ----------------------------------------------------
    if flag_col and reason_col:
        lc = check_label_consistency(df, flag_col, reason_col)
        if lc['count'] > 0:
            sev = '警告' if lc['rate'] >= 0.05 else '提示'
            r.add(sev, '标志位与原因字段自相矛盾',
                  f"{lc['count']}/{lc['n_rows']} 行（{lc['rate']:.1%}）存在矛盾："
                  f"未标记却填了原因 {lc['detail']['flag_false_but_has_reason']} 行，"
                  f"标记了却没填原因 {lc['detail']['flag_true_but_no_reason']} 行。"
                  f"比例越高，越说明该字段不是真实记录。")
        else:
            r.add('提示', '标志位与原因字段一致', '未发现矛盾组合。')
        if verbose:
            print(f"[3/5] 标签一致性：{lc['count']} 行矛盾")

    # --- 4. 极端比例 ------------------------------------------------------
    cat_cols = [c for c in df.columns
                if c != target and 2 <= df[c].nunique(dropna=True) <= max_cardinality]
    extreme = {}
    for col in cat_cols:
        found = check_extreme_ratio(df, col, target)
        if found:
            extreme[col] = found
    if extreme:
        parts = []
        for col, d in extreme.items():
            for grp, v in d.items():
                parts.append(f'{col}={grp} → {v["n_positive"]}/{v["n"]}'
                             f'（{v["rate"]:.0%}）')
        r.add('警告', '存在比例为 0% 或 100% 的分组',
              '现实数据总有反例；确定性分组通常意味着数据由规则拼装：'
              + '；'.join(parts[:6]))
    else:
        r.add('提示', '未发现 0%/100% 的确定性分组', '各分组内目标比例均有波动。')
    if verbose:
        print(f'[4/5] 极端比例：{sum(len(v) for v in extreme.values())} 组')

    # --- 5. 类别不平衡 ----------------------------------------------------
    cb = class_balance(df, target)
    if cb['n_classes'] == 2:
        base = cb['majority_baseline']
        r.add('提示', '多数类基线',
              f"目标变量分布 {cb['distribution']}，"
              f"多数类 {cb['majority_class']} 占 {base:.1%}。"
              f"任何分类模型必须显著超过 {base:.1%} 才算有效。")
    else:
        r.add('提示', '目标变量类别分布',
              f"共 {cb['n_classes']} 类，分布 {cb['distribution']}。")
    if verbose:
        print(f"[5/5] 类别不平衡：{cb['n_classes']} 类，"
              f"基线 {cb['majority_baseline']:.1%}")

    return r
