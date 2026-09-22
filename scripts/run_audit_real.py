# -*- coding: utf-8 -*-
"""Run the audit tool on the project dataset and check the known answers.

Usage (from the project root):
    .venv\\Scripts\\python.exe -m scripts.run_audit_real
or:
    $env:PYTHONPATH='.'; .venv\\Scripts\\python.exe <this file>
"""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# allow running this file directly from anywhere
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import (audit, detect_target_leakage, test_uniformity,
                       check_label_consistency, check_extreme_ratio,
                       class_balance, default_numeric_ranges)

CSV = ROOT / 'data' / 'raw' / 'smart_logistics_dataset.csv'
df = pd.read_csv(CSV, keep_default_na=False, na_values=[''])
print('loaded:', df.shape)
print()

print('=== 检测一：目标泄漏 ===')
hits = detect_target_leakage(df, 'Logistics_Delay')
for h in hits:
    print('  ', h['columns'], 'n_groups=', h['n_groups'])
combos = [set(h['columns']) for h in hits]
print('  期望含 {Shipment_Status, Traffic_Status} ->',
      'PASS' if {'Shipment_Status', 'Traffic_Status'} in combos else 'FAIL')
print(f'  共检出 {len(hits)} 组')
print()

print('=== 检测二：均匀随机性 ===')
ranges = default_numeric_ranges(df, exclude=['Logistics_Delay'])
n_uni = 0
for col, (lo, hi) in ranges.items():
    res = test_uniformity(df[col], lo, hi)
    n_uni += res['is_uniform']
    print(f"  {col:26s} ratio={res['ratio']:>8} uniform={res['is_uniform']}")
print(f'  {n_uni}/{len(ranges)} 通过 ->', 'PASS' if n_uni == len(ranges) == 9 else 'CHECK')
print()

print('=== 检测三：标签一致性 ===')
lc = check_label_consistency(df, 'Logistics_Delay', 'Logistics_Delay_Reason')
print('  ', lc)
print('  期望 count=465 ->', 'PASS' if lc['count'] == 465 else f"CHECK (got {lc['count']})")
print()

print('=== 检测四：极端比例 ===')
for col in ('Traffic_Status', 'Shipment_Status', 'Asset_ID'):
    print(f'  {col:16s}', check_extreme_ratio(df, col, 'Logistics_Delay'))
print()

print('=== 检测五：类别不平衡 ===')
print('  ', class_balance(df, 'Logistics_Delay'))
print()

print('=' * 70)
print('完整报告')
print('=' * 70)
rep = audit(df, target='Logistics_Delay',
            dataset_name='smart_logistics_dataset.csv',
            flag_col='Logistics_Delay', reason_col='Logistics_Delay_Reason',
            verbose=True)
print()
rep.print_summary()
