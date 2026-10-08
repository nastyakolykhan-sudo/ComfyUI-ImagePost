#!/usr/bin/env python3
"""Keep the vendored imgpost/ pinned to the image-post skill.

  python3 -P tools/sync_imgpost.py                 # check (default); exit 1 when anything moved
  python3 -P tools/sync_imgpost.py --update        # copy the skill's imgpost/ here and re-pin it
  python3 -P tools/sync_imgpost.py --mark-ported   # after porting a run.py change into stages.py

The pack vendors a byte-identical copy of the skill's scripts/imgpost (a ComfyUI server, local or cloud, doesn't
have the skill folder). imgpost.lock.json pins the sha256 of every vendored file and of the run.py version that
stages.py ports. --check reports: files edited here (never edit imgpost/ in the pack), skill changes not synced
yet, and run.py changes not yet ported into stages.py. After --update or a port: run tools/regress.py, commit.
"""
import argparse
import datetime
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parent.parent
LOCK = PACK / "imgpost.lock.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree(folder):
    return {p.name: sha(p) for p in sorted(Path(folder).glob("*.py"))}


def version(folder):
    m = re.search(r'__version__\s*=\s*"([^"]+)"', (Path(folder) / "__init__.py").read_text())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill-dir", help="default: the lock's skill path")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--update", action="store_true")
    g.add_argument("--mark-ported", action="store_true")
    a = ap.parse_args()

    lock = json.loads(LOCK.read_text())
    skill = Path(a.skill_dir or lock["skill"]).expanduser()
    src, dst = skill / "scripts" / "imgpost", PACK / "imgpost"
    run_py = skill / "scripts" / "run.py"

    if a.update or a.mark_ported:
        if not src.is_dir():
            sys.exit(f"skill not found: {skill}")
        if a.update:
            for p in dst.glob("*.py"):
                if not (src / p.name).exists():
                    p.unlink()
            for p in src.glob("*.py"):
                shutil.copy2(p, dst / p.name)
            lock.update(imgpost_version=version(dst), synced=datetime.date.today().isoformat(), files=tree(dst))
            print(f"synced imgpost {lock['imgpost_version']} ({len(lock['files'])} files) from {src}")
            if sha(run_py) != lock["run_py"]:
                print("run.py differs from the version stages.py ports: port the diff, then --mark-ported")
        else:
            lock["run_py"] = sha(run_py)
            print("recorded the current run.py as ported")
        LOCK.write_text(json.dumps(lock, indent=2) + "\n")
        return

    problems = []
    here = tree(dst)
    edited = sorted(n for n in set(here) | set(lock["files"]) if here.get(n) != lock["files"].get(n))
    if edited:
        problems.append("vendored imgpost/ differs from the lock (edited here?): " + ", ".join(edited))
    if src.is_dir():
        there = tree(src)
        moved = sorted(n for n in set(there) | set(lock["files"]) if there.get(n) != lock["files"].get(n))
        if moved:
            problems.append(f"the skill's imgpost/ changed since the pin ({lock['synced']}): " + ", ".join(moved) +
                            "  -> --update")
        if sha(run_py) != lock["run_py"]:
            problems.append("the skill's run.py changed since stages.py was ported -> port the diff, then --mark-ported")
    else:
        print(f"skill not found at {skill}: checked the vendored files only")
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(f"imgpost {lock['imgpost_version']} pinned {lock['synced']}: vendored copy, skill and run.py all match")


if __name__ == "__main__":
    main()
