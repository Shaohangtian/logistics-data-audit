# -*- coding: utf-8 -*-
"""
src/audit.py 的单元测试。

设计思路：用**自造的小数据集**，每个都带已知答案——
这样测的是"检测逻辑对不对"，而不是"某份数据的结果恰好是多少"。

运行（在项目根目录）：
    .venv\\Scripts\\python.exe -m pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import (  # noqa: E402
    AuditReport,
    audit,
    check_extreme_ratio,
    check_label_consistency,
    class_balance,
    default_numeric_ranges,
    detect_target_leakage,
    test_uniformity,
)

RNG = np.random.default_rng(42)
RAW = ROOT / 'data' / 'raw' / 'smart_logistics_dataset.csv'


# ===========================================================================
# 夹具：带已知答案的数据集
# ===========================================================================

@pytest.fixture
def leaky_df() -> pd.DataFrame:
    """目标变量由两个字段完全决定 —— 应被检出泄漏。"""
    n = 200
    status = RNG.choice(['Delivered', 'In Transit', 'Delayed'], n)
    traffic = RNG.choice(['Clear', 'Detour', 'Heavy'], n)
    target = ((status == 'Delayed') | (traffic == 'Heavy')).astype(int)
    return pd.DataFrame({
        'Shipment_Status': status,
        'Traffic_Status': traffic,
        'Logistics_Delay': target,
        'Noise_A': RNG.normal(0, 1, n),
    })


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """目标变量与特征独立 —— 不应检出泄漏。"""
    n = 400
    target = RNG.integers(0, 2, n)
    return pd.DataFrame({
        'Category': RNG.choice(['a', 'b', 'c'], n),
        'Group': RNG.choice(['x', 'y'], n),
        'Value': RNG.normal(50, 12, n),
        'Target': target,
    })


@pytest.fixture
def real_project_df() -> pd.DataFrame:
    """本项目真实数据（若存在）。"""
    if not RAW.exists():
        pytest.skip('data/raw/smart_logistics_dataset.csv 不存在')
    return pd.read_csv(RAW, keep_default_na=False, na_values=[''])


# ===========================================================================
# 检测一：目标泄漏
# ===========================================================================

def test_leakage_detected_by_two_field_combo(leaky_df):
    hits = detect_target_leakage(leaky_df, 'Logistics_Delay')
    combos = [set(h['columns']) for h in hits]
    assert {'Shipment_Status', 'Traffic_Status'} in combos


def test_leakage_not_reported_on_clean_data(clean_df):
    assert detect_target_leakage(clean_df, 'Target') == []


def test_leakage_single_field_is_detected():
    n = 100
    cat = RNG.choice(['p', 'q'], n)
    df = pd.DataFrame({'Cat': cat, 'Y': (cat == 'p').astype(int)})
    hits = detect_target_leakage(df, 'Y')
    assert {'Cat'} in [set(h['columns']) for h in hits]


def test_leakage_respects_exclude():
    n = 100
    cat = RNG.choice(['p', 'q'], n)
    df = pd.DataFrame({'Cat': cat, 'Y': (cat == 'p').astype(int)})
    assert detect_target_leakage(df, 'Y', exclude=['Cat']) == []


def test_leakage_missing_target_raises(clean_df):
    with pytest.raises(KeyError):
        detect_target_leakage(clean_df, 'NotAColumn')


def test_leakage_accuracy_is_one(leaky_df):
    hits = detect_target_leakage(leaky_df, 'Logistics_Delay')
    assert all(h['accuracy'] == 1.0 for h in hits)


# ===========================================================================
# 检测二：均匀随机性
# ===========================================================================

def test_uniform_series_passes():
    s = pd.Series(RNG.uniform(0, 100, 5000))
    res = test_uniformity(s, 0, 100)
    assert res['is_uniform'] is True
    assert 0.95 <= res['ratio'] <= 1.05


def test_normal_series_fails():
    s = pd.Series(RNG.normal(50, 12, 5000))
    res = test_uniformity(s, 0, 100)
    assert res['is_uniform'] is False


def test_uniformity_infers_range_when_omitted():
    s = pd.Series(RNG.uniform(0, 10, 5000))
    res = test_uniformity(s)          # 不给区间，用 min/max
    assert res['is_uniform'] is True


def test_uniformity_too_few_samples():
    res = test_uniformity(pd.Series([1.0]), 0, 10)
    assert res['is_uniform'] is False
    assert res['ratio'] is None


def test_uniformity_zero_width_range():
    res = test_uniformity(pd.Series([5.0, 5.0, 5.0]), 5, 5)
    assert res['is_uniform'] is False


def test_default_numeric_ranges_picks_numeric_only(clean_df):
    ranges = default_numeric_ranges(clean_df)
    assert 'Value' in ranges
    assert 'Category' not in ranges
    assert 'Group' not in ranges


# ===========================================================================
# 检测三：标签一致性
# ===========================================================================

def test_label_consistency_counts_both_directions():
    df = pd.DataFrame({
        'flag': [0, 0, 1, 1, 0],
        'reason': ['Weather', 'None', 'Weather', 'None', 'Traffic'],
    })
    # 行0: 未标记却有原因 -> 矛盾
    # 行1: 未标记且无原因 -> 正常
    # 行2: 标记且有原因   -> 正常
    # 行3: 标记但无原因   -> 矛盾
    # 行4: 未标记却有原因 -> 矛盾
    res = check_label_consistency(df, 'flag', 'reason')
    assert res['count'] == 3
    assert res['detail']['flag_false_but_has_reason'] == 2
    assert res['detail']['flag_true_but_no_reason'] == 1


def test_label_consistency_all_clean():
    df = pd.DataFrame({'flag': [1, 0, 1], 'reason': ['A', 'None', 'B']})
    assert check_label_consistency(df, 'flag', 'reason')['count'] == 0


def test_label_consistency_treats_none_as_no_reason():
    """字符串 'None' 必须被当作"无原因"，而不是缺失值。"""
    df = pd.DataFrame({'flag': [1], 'reason': ['None']})
    res = check_label_consistency(df, 'flag', 'reason')
    assert res['count'] == 1     # 标记了却没有原因


# ===========================================================================
# 检测四：极端比例
# ===========================================================================

def test_extreme_ratio_finds_100pct_group():
    df = pd.DataFrame({
        'grp': ['A'] * 5 + ['B'] * 5,
        'y': [1] * 5 + [1, 0, 1, 0, 1],
    })
    found = check_extreme_ratio(df, 'grp', 'y')
    assert 'A' in found and found['A']['rate'] == 1.0
    assert 'B' not in found


def test_extreme_ratio_none_when_mixed():
    df = pd.DataFrame({'grp': ['A', 'A', 'B', 'B'], 'y': [1, 0, 1, 0]})
    assert check_extreme_ratio(df, 'grp', 'y') == {}


# ===========================================================================
# 检测五：类别不平衡
# ===========================================================================

def test_class_balance_baseline():
    df = pd.DataFrame({'y': [1] * 7 + [0] * 3})
    res = class_balance(df, 'y')
    assert res['majority_baseline'] == pytest.approx(0.7)
    assert res['majority_class'] == '1'
    assert res['n_classes'] == 2


# ===========================================================================
# 报告与入口
# ===========================================================================

def test_report_add_and_count():
    r = AuditReport('d', 10, 3)
    r.add('提示', 't1', 'd1')
    r.add('严重', 't2', 'd2')
    assert r.count('严重') == 1
    assert r.has_critical is True
    assert r.sorted_findings[0]['severity'] == '严重'   # 严重排最前


def test_report_rejects_bad_severity():
    r = AuditReport('d', 1, 1)
    with pytest.raises(ValueError):
        r.add('紧急', 't', 'd')


def test_report_markdown_and_frame():
    r = AuditReport('d', 10, 3)
    r.add('警告', '标题', '细节')
    assert '# 数据质量审计报告：d' in r.to_markdown()
    assert len(r.to_frame()) == 1


def test_audit_flags_leaky_dataset(leaky_df):
    rep = audit(leaky_df, target='Logistics_Delay', dataset_name='leaky')
    assert rep.has_critical
    assert any('目标泄漏' in f['title'] for f in rep.findings)


def test_audit_passes_clean_dataset(clean_df):
    rep = audit(clean_df, target='Target', dataset_name='clean')
    assert not rep.has_critical


def test_audit_is_generic_across_schemas():
    """同一套调用必须能作用于完全不同的表结构。"""
    df1 = pd.DataFrame({'A': ['x', 'y'] * 50, 'label': [0, 1] * 50})
    df2 = pd.DataFrame({'K': RNG.normal(0, 1, 100),
                        'label': RNG.integers(0, 2, 100)})
    for df in (df1, df2):
        rep = audit(df, target='label', dataset_name='t')
        assert isinstance(rep, AuditReport)
        assert rep.n_rows == 100


# ===========================================================================
# 对真实数据的回归测试：锁住精讲文档里的数字
# ===========================================================================

def test_real_data_leakage_formula(real_project_df):
    hits = detect_target_leakage(real_project_df, 'Logistics_Delay')
    combos = [set(h['columns']) for h in hits]
    assert {'Shipment_Status', 'Traffic_Status'} in combos

    pred = ((real_project_df['Shipment_Status'] == 'Delayed')
            | (real_project_df['Traffic_Status'] == 'Heavy')).astype(int)
    assert (pred == real_project_df['Logistics_Delay']).all()   # 1000/1000


def test_real_data_heavy_is_100_percent(real_project_df):
    found = check_extreme_ratio(real_project_df, 'Traffic_Status', 'Logistics_Delay')
    assert found.get('Heavy', {}).get('n_positive') == 327
    assert found['Heavy']['rate'] == 1.0


def test_real_data_label_contradictions(real_project_df):
    res = check_label_consistency(real_project_df, 'Logistics_Delay',
                                  'Logistics_Delay_Reason')
    assert res['count'] == 465
    assert res['rate'] == pytest.approx(0.465)


def test_real_data_majority_baseline(real_project_df):
    res = class_balance(real_project_df, 'Logistics_Delay')
    assert res['majority_baseline'] == pytest.approx(0.566)


def test_real_data_has_ten_numeric_columns_nine_uniform(real_project_df):
    """数据集有 10 个非目标数值列，其中 9 个通过均匀性检验。"""
    ranges = default_numeric_ranges(real_project_df, exclude=['Logistics_Delay'])
    assert len(ranges) == 10
    ok = [c for c, (lo, hi) in ranges.items()
          if test_uniformity(real_project_df[c], lo, hi)['is_uniform']]
    assert len(ok) == 9
    # 唯一没通过的是 1..10 的离散列
    assert set(ranges) - set(ok) == {'User_Purchase_Frequency'}


def test_real_data_full_audit_is_critical(real_project_df):
    rep = audit(real_project_df, target='Logistics_Delay',
                dataset_name='smart_logistics',
                flag_col='Logistics_Delay',
                reason_col='Logistics_Delay_Reason')
    assert rep.has_critical
    assert rep.count('严重') >= 1
