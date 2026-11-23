"""Dev tool: grep a regex across all snapshots; prints which snapshots each matching line appears in.

    python tools/grep_snapshots.py REGEX [url-substring]
"""
import json
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "snapshots"
pat = re.compile(sys.argv[1], re.I)
urlf = sys.argv[2] if len(sys.argv) > 2 else ""
seen = {}
for d in sorted(root.glob("0*")):
    m = json.loads((d / "manifest.json").read_text())
    for p in m["pages"]:
        if urlf and not re.search(urlf, p["url"]):
            continue
        for line in (d / p["file"]).read_text().splitlines():
            if pat.search(line):
                seen.setdefault((p["url"], line.strip()[:300]), []).append(d.name[:2])
for (u, l), s in sorted(seen.items()):
    print(f"[{','.join(s)}] {u}\n    {l}")
