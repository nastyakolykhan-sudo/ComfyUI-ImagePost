#!/usr/bin/env python3
"""Build the template workflow in both formats from one definition and the nodes' own INPUT_TYPES/RETURN_TYPES.

  python -I tools/build_workflows.py [--job examples/demo/job.json]

Writes workflows/image-post.api.json (what tools/comfy_run.py submits) and workflows/image-post.json (what the
ComfyUI editor loads), with the job embedded in Load Job, then converts the UI file back the way the frontend
does (links by slot, widgets by position) and checks it gives the API file again. Re-run it after changing a
node's inputs or outputs. Needs a Python with torch (the pack imports it), like tools/offline_run.py.
"""
import argparse
import importlib.util
import json
import sys
import uuid
from pathlib import Path

sys.dont_write_bytecode = True
PACK = Path(__file__).resolve().parent.parent
WIDGET = {"INT", "FLOAT", "STRING", "BOOLEAN"}


def L(nid, i):
    return [str(nid), i]


def definition(job_text):
    """node id -> (class_type, title, inputs, UI position)"""
    out = "image-post/imagepost_demo_scene_refined"
    return {
        1: ("LoadImage", "Scene (generated image)", {"image": "imagepost_demo_scene.png"}, (0, 0)),
        2: ("LoadImage", "Reference (real product)", {"image": "imagepost_demo_packshot.png"}, (0, 390)),
        3: ("ImagePostLoadJob", "Load Job", {"job_json": job_text, "job_path": ""}, (0, 780)),
        4: ("ImagePostJobMasks", "Job Masks", {"job": L(3, 0), "scene": L(1, 0), "product_mask": L(6, 1)}, (470, 180)),
        5: ("ImagePostFitWarp", "Fit Warp", {"job": L(3, 0)}, (470, 0)),
        6: ("ImagePostWarpReference", "Warp Reference", {"job": L(3, 0), "homography": L(5, 0), "scene": L(1, 0),
                                                         "reference": L(2, 0), "reference_mask": L(2, 1)}, (470, 400)),
        7: ("ImagePostGradeToScene", "Grade To Scene", {"job": L(3, 0), "scene": L(1, 0), "warped": L(6, 0),
                                                        "product_mask": L(6, 1), "old_silhouette": L(4, 0),
                                                        "occluder_mask": L(4, 1), "colour": "job", "shading": "job",
                                                        "reference": L(2, 0), "coords": L(6, 3), "relight": "job"}, (900, 0)),
        8: ("ImagePostMatchFinish", "Match Finish", {"job": L(3, 0), "scene": L(1, 0), "graded": L(7, 0),
                                                     "product_mask": L(6, 1), "old_silhouette": L(4, 0),
                                                     "occluder_mask": L(4, 1)}, (1330, 260)),
        9: ("ImagePostFillLeftovers", "Fill Leftovers", {"job": L(3, 0), "scene": L(1, 0), "product_mask": L(6, 1),
                                                         "old_silhouette": L(4, 0), "occluder_mask": L(4, 1),
                                                         "coords": L(6, 3)}, (1330, 0)),
        10: ("ImagePostCompositeBehind", "Composite Behind", {"job": L(3, 0), "scene": L(1, 0), "background": L(9, 0),
                                                              "product": L(8, 0), "product_mask": L(6, 1),
                                                              "occluder_mask": L(4, 1), "ink": L(7, 4),
                                                              "coords": L(6, 3)}, (1330, 500)),
        11: ("ImagePostQASheet", "QA Sheet", {"job": L(3, 0), "before": L(1, 0), "after": L(10, 0), "reference": L(2, 0),
                                              "matte": L(10, 1), "fill_mask": L(9, 1), "reference_mask": L(2, 1),
                                              "align_report": L(5, 1), "grade_report": L(7, 2),
                                              "finish_report": L(8, 1), "fill_report": L(9, 2),
                                              "old_silhouette": L(4, 0), "occluder_mask": L(4, 1), "coords": L(6, 3)}, (1760, 0)),
        12: ("SaveImage", "Save refined image", {"images": L(10, 0), "filename_prefix": out}, (1330, 760)),
        13: ("SaveImage", "Save QA compare", {"images": L(11, 0), "filename_prefix": "image-post/qa/compare"}, (2220, 0)),
        14: ("SaveImage", "Save QA edges", {"images": L(11, 1), "filename_prefix": "image-post/qa/edges"}, (2220, 360)),
        15: ("SaveImage", "Save QA before_after", {"images": L(11, 2), "filename_prefix": "image-post/qa/before_after"}, (2220, 720)),
        16: ("SaveImage", "Save QA align_overlay", {"images": L(6, 2), "filename_prefix": "image-post/qa/align_overlay"}, (470, 700)),
        17: ("SaveImage", "Save QA grade_preview", {"images": L(7, 1), "filename_prefix": "image-post/qa/grade_preview"}, (900, 420)),
        18: ("SaveImage", "Save QA relight_map", {"images": L(7, 3), "filename_prefix": "image-post/qa/relight_map"}, (900, 780)),
        20: ("SaveImage", "Save QA audit", {"images": L(11, 4), "filename_prefix": "image-post/qa/audit"}, (2220, 1080)),
        19: ("ImagePostSavePSD", "Save PSD", {"job": L(3, 0), "scene": L(1, 0), "image": L(10, 0),
                                                           "background": L(9, 0), "fill_mask": L(9, 1), "product": L(8, 0),
                                                           "matte": L(10, 1), "filename_prefix": out}, (1760, 580)),
    }


SIZE = {"LoadImage": (400, 360), "SaveImage": (400, 330), "ImagePostLoadJob": (420, 560),
        "ImagePostQASheet": (420, 520), "ImagePostFitWarp": (380, 140), "ImagePostSavePSD": (420, 300)}


def load_pack():
    spec = importlib.util.spec_from_file_location("comfyui_imagepost", PACK / "__init__.py",
                                                  submodule_search_locations=[str(PACK)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.NODE_CLASS_MAPPINGS


def node_spec(classes, ct):
    """-> (slot inputs [(name, type)], widget names, RETURN_TYPES, RETURN_NAMES)"""
    if ct == "LoadImage":
        return [], ["image", "upload"], ("IMAGE", "MASK"), ("IMAGE", "MASK")
    if ct == "SaveImage":
        return [("images", "IMAGE")], ["filename_prefix"], (), ()
    c = classes[ct]
    it = c.INPUT_TYPES()
    slots, widgets = [], []
    for sec in ("required", "optional"):
        for name, (t, *opt) in it.get(sec, {}).items():
            o = opt[0] if opt else {}
            if (isinstance(t, list) or t in WIDGET) and not o.get("forceInput"):
                widgets.append(name)
            else:
                slots.append((name, t))
    return slots, widgets, c.RETURN_TYPES, getattr(c, "RETURN_NAMES", c.RETURN_TYPES)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", default=str(PACK / "examples" / "demo" / "job.json"))
    ap.add_argument("--name", default="image-post")
    a = ap.parse_args()
    job_text = Path(a.job).expanduser().read_text()
    json.loads(job_text)
    classes = load_pack()
    nodes_def = definition(job_text)
    api = {str(k): {"inputs": inp, "class_type": ct, "_meta": {"title": title}}
           for k, (ct, title, inp, _) in nodes_def.items()}
    (PACK / "workflows" / f"{a.name}.api.json").write_text(json.dumps(api, indent=2) + "\n")

    order, seen = [], set()

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for v in api[n]["inputs"].values():
            if isinstance(v, list):
                visit(v[0])
        order.append(n)

    for n in sorted(api, key=int):
        visit(n)
    specs = {nid: node_spec(classes, api[nid]["class_type"]) for nid in api}
    nodes, links = {}, []
    for nid in sorted(api, key=int):
        n = api[nid]
        slots, widgets, rt, rn = specs[nid]
        for name in n["inputs"]:
            if name not in widgets and name not in [s for s, _ in slots]:
                sys.exit(f"node {nid} ({n['class_type']}): {name} is not an input of the node")
        wv = [n["inputs"].get(w, "image" if w == "upload" else None) for w in widgets]
        nodes[nid] = {"id": int(nid), "type": n["class_type"], "pos": list(nodes_def[int(nid)][3]),
                      "size": list(SIZE.get(n["class_type"], (380, 220))), "flags": {}, "order": order.index(nid),
                      "mode": 0, "inputs": [{"name": s, "type": t, "link": None} for s, t in slots],
                      "outputs": [{"name": o, "type": t, "links": None, "slot_index": i}
                                  for i, (o, t) in enumerate(zip(rn, rt))],
                      "title": n["_meta"]["title"], "properties": {"Node name for S&R": n["class_type"]},
                      "widgets_values": wv}
    for nid in sorted(api, key=int):
        for slot, (name, t) in enumerate(specs[nid][0]):
            v = api[nid]["inputs"].get(name)
            if v is None:
                continue
            src, idx = v
            if specs[src][2][idx] != t:
                sys.exit(f"node {nid}.{name} expects {t}, linked to node {src} output {idx} ({specs[src][2][idx]})")
            lid = len(links) + 1
            links.append([lid, int(src), idx, int(nid), slot, t])
            nodes[nid]["inputs"][slot]["link"] = lid
            out = nodes[src]["outputs"][idx]
            out["links"] = (out["links"] or []) + [lid]
    ui = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"imagepost/{a.name}")), "revision": 0,
          "last_node_id": max(int(n) for n in api), "last_link_id": len(links),
          "nodes": [nodes[n] for n in sorted(nodes, key=int)], "links": links, "groups": [], "config": {},
          "extra": {"ds": {"scale": 0.55, "offset": [40, 40]}}, "version": 0.4}
    (PACK / "workflows" / f"{a.name}.json").write_text(json.dumps(ui, indent=1) + "\n")

    # round trip: UI -> API the way the frontend does it (links by slot, widgets by position)
    lk = {l[0]: l for l in links}
    back = {}
    for n in ui["nodes"]:
        slots, widgets, _, _ = specs[str(n["id"])]
        inputs = {w: v for w, v in zip(widgets, n["widgets_values"]) if w != "upload" and v is not None}
        for inp in n["inputs"]:
            if inp["link"] is not None:
                l = lk[inp["link"]]
                inputs[inp["name"]] = [str(l[1]), l[2]]
        back[str(n["id"])] = {"inputs": inputs, "class_type": n["type"]}
    for k, v in api.items():
        if back.get(k) != {"inputs": v["inputs"], "class_type": v["class_type"]}:
            sys.exit(f"round trip differs at node {k}")
    if set(back) != set(api):
        sys.exit("round trip differs: node ids")
    print(f"wrote workflows/{a.name}.api.json and workflows/{a.name}.json: {len(api)} nodes, {len(links)} links; "
          "UI converts back to the API file exactly")


if __name__ == "__main__":
    main()
