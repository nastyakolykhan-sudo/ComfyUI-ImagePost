#!/usr/bin/env python3
"""Execute an Image Post workflow (API format) in-process, without a ComfyUI server. For tests and regression.

  python -I tools/offline_run.py --workflow workflows/image-post.api.json \\
      --scene path/to/scene.png --reference path/to/packshot.png \\
      [--job JOB.json] [--stage align|grade|all] [--psd] --out ./ip-offline

Takes the same arguments as comfy_run.py and writes the same --out layout. It imports this pack, stands in for
the core nodes the workflows use (LoadImage, SaveImage, PreviewImage, MaskToImage) and ComfyUI's folder_paths
with ComfyUI's own pixel conversions, checks inputs and link types the way ComfyUI validates a prompt, and runs
the output nodes' dependencies in order. Needs a Python with torch, numpy, scipy and Pillow (ComfyUI's own works).
"""
import argparse
import importlib.util
import json
import sys
import tempfile
import time
import types
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import comfy_run as cr

PACK = Path(__file__).resolve().parent.parent


def load_pack():
    spec = importlib.util.spec_from_file_location("comfyui_imagepost", PACK / "__init__.py",
                                                  submodule_search_locations=[str(PACK)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.NODE_CLASS_MAPPINGS


# ---- stand-ins for ComfyUI core nodes (same conversions as ComfyUI's nodes.py) ----

class LoadImage:
    paths = {}   # image name -> local file

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("STRING", {})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image):
        img = Image.open(self.paths[image])
        output_images, output_masks, w, h = [], [], None, None
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            rgb = i.convert("RGB")
            if not output_images:
                w, h = rgb.size
            if rgb.size != (w, h):
                continue
            t = torch.from_numpy(np.array(rgb).astype(np.float32) / 255.0)[None,]
            if "A" in i.getbands():
                mask = 1. - torch.from_numpy(np.array(i.getchannel("A")).astype(np.float32) / 255.0)
            elif i.mode == "P" and "transparency" in i.info:
                mask = 1. - torch.from_numpy(np.array(i.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(t)
            output_masks.append(mask.unsqueeze(0))
        if len(output_images) > 1 and img.format not in ("MPO",):
            return torch.cat(output_images, dim=0), torch.cat(output_masks, dim=0)
        return output_images[0], output_masks[0]


class SaveImage:
    out_dir = Path(".")
    type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "filename_prefix": ("STRING", {"default": "ComfyUI"})}}

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True

    def save_images(self, images, filename_prefix="ComfyUI"):
        sub, _, stem = filename_prefix.rpartition("/")
        folder = self.out_dir / self.type / sub
        folder.mkdir(parents=True, exist_ok=True)
        counter = 1 + max([int(p.stem.split("_")[-2]) for p in folder.glob(f"{stem}_*_.png")] or [0])
        results = []
        for image in images:
            i = 255. * image.cpu().numpy()
            name = f"{stem}_{counter:05}_.png"
            Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).save(folder / name, compress_level=4)
            results.append({"filename": name, "subfolder": sub, "type": self.type})
            counter += 1
        return {"ui": {"images": results}}


class PreviewImage(SaveImage):
    type = "temp"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    def save_images(self, images, filename_prefix="ComfyUI_temp"):
        return super().save_images(images, filename_prefix)


class MaskToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mask_to_image"

    def mask_to_image(self, mask):
        return (mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3),)


CORE = {"LoadImage": LoadImage, "SaveImage": SaveImage, "PreviewImage": PreviewImage, "MaskToImage": MaskToImage}

# ComfyUI's folder_paths, for nodes that write files themselves (Save PSD): the same output/ and temp/ folders
folder_paths = types.ModuleType("folder_paths")
folder_paths.get_output_directory = lambda: str(SaveImage.out_dir / "output")
folder_paths.get_temp_directory = lambda: str(SaveImage.out_dir / "temp")
sys.modules.setdefault("folder_paths", folder_paths)


# ---- validation and execution ----

def validate(wf, classes):
    errors = []
    for nid, node in wf.items():
        cls = classes.get(node["class_type"])
        if cls is None:
            errors.append(f"node {nid}: unknown class_type {node['class_type']}")
            continue
        spec = cls.INPUT_TYPES()
        declared = {**spec.get("required", {}), **spec.get("optional", {})}
        for name in spec.get("required", {}):
            if name not in node["inputs"]:
                errors.append(f"node {nid} ({node['class_type']}): required input {name} missing")
        for name, value in node["inputs"].items():
            if name not in declared:
                errors.append(f"node {nid} ({node['class_type']}): unknown input {name}")
                continue
            want = declared[name][0]
            if cr.is_link(value):
                src = wf.get(value[0])
                if src is None:
                    errors.append(f"node {nid}.{name}: links to missing node {value[0]}")
                    continue
                rt = classes[src["class_type"]].RETURN_TYPES
                got = rt[value[1]] if value[1] < len(rt) else None
                if got != want:
                    errors.append(f"node {nid}.{name}: expects {want}, linked to {src['class_type']} output {value[1]} ({got})")
            elif isinstance(want, list) and value not in want:
                errors.append(f"node {nid}.{name}: {value!r} not in {want}")
            elif want == "STRING" and not isinstance(value, str):
                errors.append(f"node {nid}.{name}: expects a string")
    return errors


def execute(wf, classes):
    """-> history-style entry {"outputs": {node id: ui}, "status": {...}}"""
    outputs = [nid for nid, n in wf.items() if getattr(classes[n["class_type"]], "OUTPUT_NODE", False)]
    order, seen = [], set()

    def visit(nid):
        if nid in seen:
            return
        seen.add(nid)
        for v in wf[nid]["inputs"].values():
            if cr.is_link(v):
                visit(v[0])
        order.append(nid)

    for nid in sorted(outputs, key=int):
        visit(nid)
    results, ui = {}, {}
    for nid in order:
        node = wf[nid]
        cls = classes[node["class_type"]]
        kwargs = {k: (results[v[0]][v[1]] if cr.is_link(v) else v) for k, v in node["inputs"].items()}
        t0 = time.time()
        try:
            ret = getattr(cls(), cls.FUNCTION)(**kwargs)
        except Exception as e:
            return {"outputs": ui, "status": {"status_str": "error", "completed": False, "messages": [
                ["execution_error", {"node_id": nid, "node_type": node["class_type"],
                                     "exception_message": str(e), "exception_type": type(e).__name__}]]}}
        if isinstance(ret, dict):
            if ret.get("ui"):
                ui[nid] = ret["ui"]
            ret = ret.get("result", ())
        results[nid] = ret
        print(f"  ran {nid:>3} {node['class_type']:28s} {time.time() - t0:6.2f} s")
    return {"outputs": ui, "status": {"status_str": "success", "completed": True, "messages": []}}


def run(workflow, out, scene=None, reference=None, images=(), job=None, stage="all", psd=False):
    """Offline equivalent of comfy_run.py -> history entry (also written to out/history.json)."""
    wf = cr.load_workflow(workflow)
    if job:
        cr.set_job(wf, job)
    targets = cr.image_targets(wf, scene, reference, images)
    wf = cr.prune(cr.keep_psd(wf, cr.psd_wanted(wf, psd)), stage)
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    classes = {**CORE, **load_pack()}
    for nid, node in wf.items():
        if node["class_type"] == "LoadImage":
            if nid not in targets:
                sys.exit(f"Load Image node {nid} ({node['inputs']['image']}): give its file with --scene, --reference or --image {nid}=PATH")
            node["inputs"]["image"] = targets[nid].name
            LoadImage.paths[targets[nid].name] = targets[nid]
    errors = validate(wf, classes)
    if errors:
        sys.exit("prompt rejected:\n  " + "\n  ".join(errors))
    (out / "prompt.json").write_text(json.dumps(wf, indent=1))
    with tempfile.TemporaryDirectory() as server:   # stands in for ComfyUI's output/ and temp/ folders
        SaveImage.out_dir = Path(server)
        entry = execute(wf, classes)

        def fetch(img):
            return (SaveImage.out_dir / img["type"] / img["subfolder"] / img["filename"]).read_bytes()

        written = cr.save_outputs(entry, wf, fetch, out)
    return entry, written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--scene")
    ap.add_argument("--reference")
    ap.add_argument("--image", action="append", default=[], metavar="NODE_ID=PATH")
    ap.add_argument("--job")
    ap.add_argument("--stage", choices=("align", "grade", "all"), default="all")
    ap.add_argument("--psd", action="store_true", help="also run the Save PSD nodes (same as \"psd\": true in the job)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)   # keep progress and errors in order when piped
    t0 = time.time()
    entry, written = run(a.workflow, a.out, a.scene, a.reference, a.image, a.job, a.stage, a.psd)
    status = entry["status"]
    if status["status_str"] == "error":
        msg = status["messages"][0][1]
        sys.exit(f"failed     node {msg['node_id']} ({msg['node_type']}): {msg['exception_message']}")
    print(f"done       {time.time() - t0:.1f} s")
    for p in written:
        print(f"saved      {p}")
    cr.summary(Path(a.out).expanduser())


if __name__ == "__main__":
    main()
