# -*- coding: utf-8 -*-
"""生成 README 与审计报告引用的图，输出到 reports/figures/。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / 'reports' / 'figures'
FIGS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / 'data' / 'raw' / 'smart_logistics_dataset.csv',
                 keep_default_na=False, na_values=[''])
df['Logistics_Delay'] = df['Logistics_Delay'].astype(int)

# 图 1：泄漏证据 —— 各(路况, 运输状态)组合的延迟比例
rate = df.groupby(['Traffic_Status', 'Shipment_Status'],
                  observed=True)['Logistics_Delay'].mean().unstack()
plt.figure(figsize=(9, 5))
sns.heatmap(rate, annot=True, fmt='.0%', cmap='RdBu_r', vmin=0, vmax=1,
            linewidths=0.5, linecolor='lightgrey',
            cbar_kws={'label': '延迟比例'})
plt.title('各(路况, 运输状态)组合的延迟比例\n每个格子只有 0% 或 100%，无一反例')
plt.tight_layout()
plt.savefig(FIGS / 'leakage_heatmap.png', dpi=150)
plt.close()

# 图 2：均匀随机性 —— 实际标准差 / 理论标准差
import numpy as np
RANGES = {'Latitude': (-90, 90), 'Longitude': (-180, 180),
          'Temperature': (18, 30), 'Humidity': (50, 80),
          'Inventory_Level': (100, 500), 'Asset_Utilization': (60, 100),
          'User_Transaction_Amount': (100, 500), 'Demand_Forecast': (100, 300),
          'Waiting_Time': (10, 60)}
rows = [{'field': c, 'ratio': df[c].std(ddof=1) / ((hi - lo) / np.sqrt(12))}
        for c, (lo, hi) in RANGES.items()]
u = pd.DataFrame(rows).sort_values('ratio')

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(u['field'], u['ratio'], color='steelblue')
ax.axvline(1.0, color='crimson', linestyle='--', label='理论值 1.0')
ax.axvspan(0.9, 1.1, color='green', alpha=0.08, label='±10% 判定区间')
ax.set_xlim(0, 1.2)
ax.set_xlabel('实际标准差 / 均匀分布理论标准差')
ax.set_title('9 个数值字段全部贴近均匀分布理论值')
ax.legend()
plt.tight_layout()
plt.savefig(FIGS / 'uniformity_ratios.png', dpi=150)
plt.close()

print('figures written to', FIGS)
for f in sorted(FIGS.glob('*.png')):
    print('  ', f.name, f.stat().st_size, 'bytes')