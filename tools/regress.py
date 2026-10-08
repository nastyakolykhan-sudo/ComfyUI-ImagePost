#!/usr/bin/env python3
"""Regression: the node pack against the image-post engine's own run.py, on the same job and images.

  python -I tools/regress.py                          # the demo job, pack run in-process (needs torch)
  python3 -P tools/regress.py --server http://127.0.0.1:8188 [--header "K: V"]   # pack run on a ComfyUI server
  python -I tools/regress.py --job JOB.json [--scene PATH] [--reference PATH] [--workflow WF.api.json] [--psd]

Runs the engine's scripts/run.py (--skill-dir, default ~/Documents/Claude/image-post) with --job-dir and --output
inside --work, never at the job's own output path; or pass --skill-out to compare with an existing engine output.
Prints the pixel differences (max, count above 0, 1 and --levels) and which report sections match; with --psd
both sides also write the layered PSD and their psd reports are compared.
Exit 0 when fewer than --max-px pixels differ by more than --levels, nothing outside the work box differs and
(with --psd) the PSDs have the same layers and flatten to their PNGs within 2 levels.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image

import comfy_run as cr

PACK = Path(__file__).resolve().parent.parent
SKILL = Path("~/Documents/Claude/image-post").expanduser()


def resolve(p, base):
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else Path(base) / p


def result_image(out, wf):
    """The Save Image output fed by Composite Behind's image."""
    entry = json.loads((out / "history.json").read_text())
    for nid, node in wf.items():
        src = node["inputs"].get("images")
        if node["class_type"] == "SaveImage" and cr.is_link(src) and \
                wf[src[0]]["class_type"] == "ImagePostCompositeBehind" and src[1] == 0:
            img = entry["outputs"][nid]["images"][0]
            return out / img.get("subfolder", "") / img["filename"]
    sys.exit("no Save Image node takes Composite Behind's image")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill-dir", default=str(SKILL))
    ap.add_argument("--job", help="default: this pack's examples/demo/job.json")
    ap.add_argument("--scene", help="default: the job's scene")
    ap.add_argument("--reference", help="default: the job's reference")
    ap.add_argument("--workflow", default=str(PACK / "workflows" / "image-post.api.json"))
    ap.add_argument("--psd", action="store_true", help="also write and compare the layered PSDs")
    ap.add_argument("--server", help="run the pack on this ComfyUI server instead of in-process")
    ap.add_argument("--header", action="append", default=[])
    ap.add_argument("--skill-python", default=sys.executable)
    ap.add_argument("--skill-out", help="existing skill output to compare with (skips the skill run)")
    ap.add_argument("--work", help="scratch folder (default: a new temp folder)")
    ap.add_argument("--levels", type=int, default=8)
    ap.add_argument("--max-px", type=int, default=20)
    a = ap.parse_args()

    skill = Path(a.skill_dir).expanduser()
    job_p = Path(a.job).expanduser().resolve() if a.job else PACK / "examples" / "demo" / "job.json"
    job = json.loads(job_p.read_text())
    scene = Path(a.scene).expanduser() if a.scene else resolve(job["scene"], job_p.parent)
    reference = Path(a.reference).expanduser() if a.reference else resolve(job["reference"], job_p.parent)
    work = Path(a.work).expanduser().resolve() if a.work else Path(tempfile.mkdtemp(prefix="imagepost-regress-"))
    work.mkdir(parents=True, exist_ok=True)

    # 1. the skill
    if a.skill_out:
        skill_out, skill_rep = Path(a.skill_out).expanduser(), None
    else:
        skill_out = work / "skill" / "out.png"   # always inside --work: never the job's own output path
        cmd = [a.skill_python, "-P", str(skill / "scripts" / "run.py"), str(job_p), "--job-dir", str(work / "skill"),
               "--output", str(skill_out), "--scene", str(scene)] + (["--psd"] if a.psd else [])
        print("skill      " + " ".join(cmd[2:3]) + " ...")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"skill run failed:\n{r.stdout}{r.stderr}")
        skill_rep = json.loads((work / "skill" / "report.json").read_text())

    # 2. the pack, through the same API workflow an agent would submit
    out = work / "pack"
    args = ["--workflow", a.workflow, "--job", str(job_p), "--scene", str(scene), "--reference", str(reference),
            "--out", str(out)] + (["--psd"] if a.psd else [])
    if a.server:
        cmd = [sys.executable, "-P", str(PACK / "tools" / "comfy_run.py"), "--server", a.server] + args
        for h in a.header:
            cmd += ["--header", h]
        print(f"pack       {a.server}")
        r = subprocess.run(cmd, capture_output=True, text=True)
    else:
        cmd = [sys.executable, "-I", str(PACK / "tools" / "offline_run.py")] + args
        print("pack       in-process (offline_run.py)")
        r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"pack run failed:\n{r.stdout}{r.stderr}")
    wf = json.loads((out / "prompt.json").read_text())
    pack_out = result_image(out, wf)
    pack_rep = json.loads((out / "report.json").read_text())

    # 3. compare
    A = np.asarray(Image.open(pack_out).convert("RGB")).astype(np.int16)
    B = np.asarray(Image.open(skill_out).convert("RGB")).astype(np.int16)
    if A.shape != B.shape:
        sys.exit(f"size mismatch: pack {A.shape} vs skill {B.shape}")
    d = np.abs(A - B).max(2)
    x0, y0, x1, y1 = pack_rep["roi"]
    inside = np.zeros(d.shape, bool)
    inside[y0:y1, x0:x1] = True
    res = {"pack_output": str(pack_out), "skill_output": str(skill_out), "max_level_diff": int(d.max()),
           "px_diff_gt0": int((d > 0).sum()), "px_diff_gt1": int((d > 1).sum()),
           f"px_diff_gt{a.levels}": int((d > a.levels).sum()), "px_diff_outside_roi": int((d[~inside] > 0).sum()),
           "pack_outside_roi_changed": pack_rep["change"]["outside_roi_changed"]}
    if skill_rep:
        res["report_sections_identical"] = {k: pack_rep.get(k) == skill_rep.get(k)
                                            for k in ("align", "grade", "finish", "fill", "change")}
    psd_ok = True
    if a.psd:
        texts = sorted(out.glob("ImagePostSavePSD-*.txt"))
        pack_psd = json.loads(texts[0].read_text())["psd"] if texts else None
        skill_psd = skill_rep.get("psd") if skill_rep else None
        res["psd"] = {"pack": pack_psd and {k: pack_psd[k] for k in ("layers", "layers_vs_png")},
                      "skill": skill_psd and {k: skill_psd[k] for k in ("layers", "layers_vs_png")}}
        psd_ok = bool(pack_psd) and pack_psd["layers_vs_png"]["max_levels"] <= 2 and \
            (skill_psd is None or pack_psd["layers"] == skill_psd["layers"])
        res["psd"]["pass"] = psd_ok
    ok = res[f"px_diff_gt{a.levels}"] < a.max_px and res["px_diff_outside_roi"] == 0 and \
        res["pack_outside_roi_changed"] == 0 and psd_ok
    res["pass"] = bool(ok)
    (work / "regress.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"{'PASS' if ok else 'FAIL'}: {res[f'px_diff_gt{a.levels}']} px differ by more than {a.levels} levels "
          f"(limit {a.max_px}); work folder {work}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
