"""Record or verify the public API signature snapshot (no name may disappear)."""
import inspect
import json
import sys
from pathlib import Path

import SKSurrogate

SNAP = Path(__file__).with_name("api_snapshot.json")


def collect():
    out = {}
    for name in SKSurrogate.__all__:
        obj = getattr(SKSurrogate, name)
        try:
            target = obj.__init__ if inspect.isclass(obj) else obj
            sig = list(inspect.signature(target).parameters)
        except (TypeError, ValueError):
            sig = []
        members = (
            sorted(m for m in dir(obj) if not m.startswith("_")) if inspect.isclass(obj) else []
        )
        out[name] = {"params": [p for p in sig if p != "self"], "members": members}
    return out


if __name__ == "__main__":
    cur = collect()
    if "--write" in sys.argv:
        SNAP.write_text(json.dumps(cur, indent=1, sort_keys=True))
        print("written", len(cur))
        sys.exit(0)
    old = json.loads(SNAP.read_text())
    bad = []
    for n, v in old.items():
        if n not in cur:
            bad.append(f"missing name {n}")
            continue
        for k in ("params", "members"):
            miss = set(v[k]) - set(cur[n][k])
            if miss:
                bad.append(f"{n}.{k} missing {sorted(miss)}")
    print("\n".join(bad) or "OK: nothing removed")
    sys.exit(1 if bad else 0)
