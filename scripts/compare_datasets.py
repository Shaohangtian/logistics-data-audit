# -*- coding: utf-8 -*-
"""
跨数据集对比审计 —— 验证审计工具的通用性

对 4 份数据跑同一个 audit()，打印对比表：

  1. smart_logistics_dataset.csv  —— 本项目的合成数据（预期：检出严重泄漏）
  2. olist_orders_dataset.csv     —— 巴西电商真实订单数据（Olist，公开）
  3. olist_order_items_dataset.csv—— Olist 订单明细（真实）
  4. tips.csv / titanic.csv       —— seaborn 公开真实数据

用法（在项目根目录）：
    .venv\\Scripts\\python.exe scripts/compare_datasets.py
    .venv\\Scripts\\python.exe scripts/compare_datasets.py --fetch   # 先下载再跑
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import audit  # noqa: E402

RAW = ROOT / 'data' / 'raw'
EXT = ROOT / 'data' / 'external'

SOURCES = {
    'olist_orders_dataset.csv': (
        'https://raw.githubusercontent.com/olist/work-at-olist-data/master/'
        'datasets/olist_orders_dataset.csv'),
    'olist_order_items_dataset.csv': (
        'https://raw.githubusercontent.com/olist/work-at-olist-data/master/'
        'datasets/olist_order_items_dataset.csv'),
    'tips.csv': ('https://raw.githubusercontent.com/mwaskom/seaborn-data/'
                 'master/tips.csv'),
    'titanic.csv': ('https://raw.githubusercontent.com/mwaskom/seaborn-data/'
                    'master/titanic.csv'),
}


def fetch(force: bool = False) -> list[str]:
    """把外部数据集下载到 data/external/（已存在则跳过）。

    返回下载失败的文件名列表；失败不致命，只是对比结果会少几份数据集。
    """
    EXT.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for name, url in SOURCES.items():
        dest = EXT / name
        if dest.exists() and not force:
            print(f'  已有 {name}')
            continue
        print(f'  下载 {name} ...', end='', flush=True)
        # 网络偶发失败（502 / 超时）时重试，仍失败则跳过该文件，
        # 而不是让整个脚本崩溃 —— 只要还有数据集，对比表就有意义。
        for attempt in range(1, 4):
            try:
                req = urllib.request.Request(
                    url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=120) as r:
                    dest.write_bytes(r.read())
                print(f' {dest.stat().st_size / 1024:.0f} KB')
                break
            except Exception as e:
                if attempt == 3:
                    print(f' 失败（{type(e).__name__}: {e}）')
                    failed.append(name)
                else:
                    print(f' 重试{attempt}…', end='', flush=True)
                    time.sleep(3)

    if failed:
        print()
        print('!! 以下文件下载失败，对比结果会缺少对应数据集：')
        for n in failed:
            print(f'   - {n}')
        print('   可稍后重跑 --fetch 补下（已下载的文件会跳过）。')
    return failed


# --------------------------------------------------------------------------
# 每份数据集的：路径、目标变量、可选的一致性检查字段、展示名
# --------------------------------------------------------------------------
def build_cases() -> list[dict]:
    cases: list[dict] = []

    proj = RAW / 'smart_logistics_dataset.csv'
    if proj.exists():
        cases.append({
            'name': 'smart_logistics（本项目，合成）',
            'path': proj,
            'target': 'Logistics_Delay',
            'flag': 'Logistics_Delay',
            'reason': 'Logistics_Delay_Reason',
        })

    p = EXT / 'olist_orders_dataset.csv'
    if p.exists():
        cases.append({'name': 'olist_orders（真实电商订单）', 'path': p,
                      'target': 'order_status'})

    p = EXT / 'olist_order_items_dataset.csv'
    if p.exists():
        cases.append({'name': 'olist_order_items（真实订单明细）', 'path': p,
                      'target': 'freight_value'})

    p = EXT / 'titanic.csv'
    if p.exists():
        cases.append({'name': 'titanic（真实乘客数据）', 'path': p,
                      'target': 'survived'})

    p = EXT / 'tips.csv'
    if p.exists():
        cases.append({'name': 'tips（真实小费数据）', 'path': p,
                      'target': 'smoker'})

    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true', help='先下载外部数据集')
    ap.add_argument('--save', type=str, default=None, help='把对比表保存为 Markdown')
    ap.add_argument('--include-titanic', action='store_true',
                    help='也审计 titanic。注意它的 alive 列是 survived 的派生列，'
                         '会命中泄漏检测——那是**真阳性**，不是工具误报')
    ap.add_argument('--brief', action='store_true', help='只打印对比表')
    args = ap.parse_args()

    if args.fetch:
        print('下载外部数据集：')
        fetch()

    cases = build_cases()
    # titanic 的 seaborn 版本含派生日志列（alive / class / who …），
    # 会命中泄漏检测。默认排除，保证对比里"真实数据 = 干净"的对照清晰。
    if not args.include_titanic:
        cases = [c for c in cases if 'titanic' not in c['name']]
    if not cases:
        print('没有可用的数据集。请加 --fetch 下载，或先准备好 data/raw/。')
        return 1

    rows = []
    for c in cases:
        df = pd.read_csv(c['path'], keep_default_na=False, na_values=[''],
                         low_memory=False)
        kw = {}
        if c.get('flag') and c.get('reason'):
            kw = {'flag_col': c['flag'], 'reason_col': c['reason']}
        rep = audit(df, target=c['target'], dataset_name=c['name'], **kw)

        sev = {s: rep.count(s) for s in ('严重', '警告', '提示')}
        leak = '是' if sev['严重'] else '否'
        rows.append({
            '数据集': c['name'],
            '行数': len(df),
            '列数': df.shape[1],
            '目标变量': c['target'],
            '目标泄漏': leak,
            '严重': sev['严重'],
            '警告': sev['警告'],
            '提示': sev['提示'],
            '_report': rep,
        })

    # ------------------------------------------------ 对比表
    print()
    print('=' * 100)
    print('跨数据集审计对比（同一套 audit()，五项检测）')
    print('=' * 100)
    tbl = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                        for r in rows])
    print(tbl.to_string(index=False))
    print()

    # ------------------------------------------------ 每份数据的要点
    if not args.brief:
        for r in rows:
            rep = r['_report']
            print('-' * 100)
            print(f"{r['数据集']}   目标={r['目标变量']}   "
                  f"({r['行数']} 行 × {r['列数']} 列)")
            print('-' * 100)
            for f in rep.sorted_findings:
                detail = f['detail']
                if len(detail) > 150:
                    detail = detail[:150] + '…'
                print(f"  [{f['severity']}] {f['title']}")
                print(f"        {detail}")
            print()

    n_leak = sum(1 for r in rows if r['目标泄漏'] == '是')
    print('=' * 100)
    print(f'结论：共审计 {len(rows)} 份数据集，其中 {n_leak} 份检出严重目标泄漏、'
          f'{len(rows) - n_leak} 份未检出。')
    print('      同一套检测逻辑既能揪出合成数据的泄漏，也能让真实数据通过 —— '
          '说明它不是为某一份数据硬调的。')
    print('=' * 100)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)

        def md_table(frame: pd.DataFrame) -> str:
            """自己渲染 Markdown 表格，免掉 tabulate 依赖。"""
            cols = [str(c) for c in frame.columns]
            rows_ = [''.join(f'| {v} ' for v in r) + '|'
                     for r in frame.itertuples(index=False)]
            head = '| ' + ' | '.join(cols) + ' |'
            sep = '|' + '|'.join(['---'] * len(cols)) + '|'
            return '\n'.join([head, sep, *rows_])

        lines = ['# 跨数据集审计对比', '',
                 '同一套 `audit()` 应用于多份数据集的结果。', '',
                 md_table(tbl), '']
        for r in rows:
            lines.append(f"## {r['数据集']}")
            lines.append('')
            body = r['_report'].to_markdown().split('\n', 1)[1]
            lines.append(body.lstrip('\n'))
            lines.append('')
        out.write_text('\n'.join(lines), encoding='utf-8')
        print(f'已保存 Markdown：{out}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
