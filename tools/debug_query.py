"""Dev tool: show the top-ranked chunks and the engine's answer for a question.

    python tools/debug_query.py "question" [as_of]
"""
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "solution")]
from run_eval import load_snapshots  # noqa: E402
from cuknow.engine import Engine  # noqa: E402
from cuknow.query import analyze  # noqa: E402
from cuknow.store import Store  # noqa: E402

st = Store()
for s in load_snapshots(ROOT / "snapshots"):
    st.add_snapshot(s)
st.build()
eng = Engine(st)
q = sys.argv[1]
as_of = sys.argv[2] if len(sys.argv) > 2 else None
qi = analyze(q)
print({k: v for k, v in qi.__dict__.items() if v and k not in ("raw", "norm")})
snap = eng.view(as_of, qi)
sc = eng.scores(qi, snap)
for cid in eng.top(sc, 12):
    r = st.chunks[cid]
    print(f"{sc[cid]:7.2f} {r.chunk.kind:5} {st.page_of(r).ptype:10} {r.url[23:90]:68} | {r.chunk.text[:110]}")
t = time.time()
res = eng.answer(q, as_of)
print(f"\n[{res.kind}] ({time.time()-t:.3f}s) {res.text}\n  sources={[u for u,_,_ in res.sources]}\n  note={res.conflict_note}")
