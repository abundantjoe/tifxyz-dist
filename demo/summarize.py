"""Summarise demo/pairwise.py output as markdown. Usage: python demo/summarize.py pairs.jsonl [VOXEL_UM]"""
import json, sys
from collections import Counter

R = [json.loads(l) for l in open(sys.argv[1])]
um = float(sys.argv[2]) if len(sys.argv) > 2 else None
def short(n):
    s = n.split('-', 1)[-1].replace('_flatboi', '')
    return s.split('_')[0] if s.startswith('w') else 'auto_grown_' + s.split('_')[-1]
print(f'pairs: {len(R)}; total compute {sum(r["seconds"] for r in R) / 60:.1f} min '
      f'(median {sorted(r["seconds"] for r in R)[len(R) // 2]:.1f} s/pair, unrefined)\n')
gap = [r['upper'] - r['lower'] for r in R if r.get('upper') is not None]
print(f'interval width (upper - lower): median {sorted(gap)[len(gap) // 2]:.2f} vox, max {max(gap):.2f} vox\n')
print('| margin (vox) | ' + (' margin (um) |' if um else '') + ' PASS | FAIL | INCONCLUSIVE |')
print('|---|' + ('---|' if um else '') + '---|---|---|')
for m in (1, 5, 20, 50, 100):
    c = Counter('PASS' if r['lower'] >= m else ('FAIL' if r.get('upper') is not None and r['upper'] < m else 'INCONCLUSIVE') for r in R)
    print(f'| {m} | ' + (f'{m * um:.0f} |' if um else '') + f' {c["PASS"]} | {c["FAIL"]} | {c["INCONCLUSIVE"]} |')
print('\nPairs with a witness closer than 1 vox (surfaces touch or cross):\n')
print('| segment | prediction segment | lower | upper | witness (x, y, z) |')
print('|---|---|---|---|---|')
for r in sorted((r for r in R if r.get('upper') is not None and r['upper'] < 1), key=lambda r: (r['a'], r['b'])):
    w = ', '.join(f'{v:.1f}' for v in r['witness'])
    print(f'| {short(r["a"])} | {short(r["b"])} | {r["lower"]:.2f} | {r["upper"]:.3f} | {w} |')
