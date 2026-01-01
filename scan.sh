while true; do
  echo "=== $(date) ==="
  python - <<'PY'
import csv,glob
files = sorted(glob.glob('scan_out/sample_*.csv'))
rows = []
for f in files:
    try:
        with open(f,newline='') as fh:
            r = csv.DictReader(fh)
            for rec in r:
                w=int(rec.get('wins',0)); l=int(rec.get('losses',0)); d=int(rec.get('draws',0))
                total=w+l+d
                if total==0: continue
                score=(w+0.5*d)/total
                rows.append((score,w,l,d,total,f))
    except Exception:
        continue
rows.sort(reverse=True, key=lambda x: x[0])
if not rows:
    print("no results yet in scan_out/")
else:
    print(f"{'score':>6}  {'wins':>4} {'losses':>6} {'draws':>5} {'total':>5}  file")
    for s,w,l,d,t,f in rows[:10]:
        print(f"{s:6.3f}  {w:4d} {l:6d} {d:5d} {t:5d}  {f}")
PY
  sleep 10
done
