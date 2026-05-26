"""Trace q03 (binary search) and q04 (stack/queue) across all three runs."""
import json

A = json.load(open('results/evaluation_20260523_121048.json'))
B = json.load(open('results/evaluation_v1_cohere_fixed.json'))
C = json.load(open('results/evaluation_final.json'))


def fmt(r):
    if r.get('fallback'):
        return "fallback"
    return f"deliver final={r['final_score']:.4f} style={r['style_score']:.4f} gr={r['groundedness_score']:.4f} conf={r['confidence_score']:.4f}"


print("v1 May-23 (Cohere broken) — q03 binary search & q04 stack/queue:")
for r in A:
    if r['id'] in ('q03', 'q04'):
        print(f"  {r['id']}/{r['leader']}: {fmt(r)}")
print()
print("v1 + Cohere (control) — q03 binary search & q04 stack/queue:")
for r in B:
    if r['id'] in ('q03', 'q04'):
        print(f"  {r['id']}/{r['leader']}: {fmt(r)}")
print()
print("v2 final — q12 binary search & q13 stack/queue:")
for r in C:
    if r['id'] in ('q12', 'q13'):
        print(f"  {r['id']}/{r['leader']}: {fmt(r)}")
