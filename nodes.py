"""Image Post nodes: deterministic product compositing (the image-post engine) as ComfyUI nodes.

Data conventions
- `scene` is the full generated frame. The job's roi is the work box; nothing outside it changes.
- Every other IMAGE and MASK passed between Image Post nodes covers the work box only. Mask inputs also take
  a scene-sized mask (mask editor, segmentation) and crop it to the work box.
- Product layers (warped, graded, product) are sRGB like any IMAGE but keep highlight headroom above 1.0,
  which Match Finish rolls off. Don't clamp them in between.
- Conversion happens here, at the node boundary; stages.py runs the skill's own imgpost code in float64.
"""
import hashlib
import json
import os
import re

import numpy as np
import torch

JOB = "IMAGEPOST_JOB"
HOMOGRAPHY = "IMAGEPOST_HOMOGRAPHY"
COORDS = "IMAGEPOST_COORDS"   # reference coordinates of every work-box pixel, float64 (grade.shading reads them)
INK = "IMAGEPOST_INK"         # the grade's white field and ink colour, for the job's matte 'ink' (None otherwise)
CATEGORY = "image/image-post"

COLOUR_CHOICES = ["job", "scene", "reference"]
SHADING_CHOICES = ["job", "off"]
RELIGHT_CHOICES = ["job", "off"]


def _stages():
    from . import stages   # imgpost pulls in scipy: load it on first use, not at ComfyUI startup
    return stages


# ---- node boundary: ComfyUI tensors <-> the arrays run.py works on ----

def _frame(image, name):
    """IMAGE [1, H, W, C] -> float32 H x W x 3."""
    if image.dim() != 4:
        raise ValueError(f"{name}: expected an IMAGE [B, H, W, C], got shape {tuple(image.shape)}")
    if image.shape[0] != 1:
        raise ValueError(f"{name}: Image Post works on one image per job, got a batch of {image.shape[0]}")
    a = image[0].detach().to("cpu", torch.float32).numpy()
    return np.repeat(a, 3, 2) if a.shape[2] == 1 else a[..., :3]


def _u8(image, name):
    """IMAGE -> uint8. Recovers the file's exact values from LoadImage's x / 255."""
    return np.clip(np.round(_frame(image, name).astype(np.float64) * 255), 0, 255).astype(np.uint8)


def _box_view(arr, box, scene_hw, name):
    """Work-box-sized arrays pass through; scene-sized ones are cropped to the work box."""
    x0, y0, x1, y1 = box
    if arr.shape[:2] == (y1 - y0, x1 - x0):
        return arr
    if arr.shape[:2] == tuple(scene_hw):
        return arr[y0:y1, x0:x1]
    raise ValueError(f"{name} is {arr.shape[1]}x{arr.shape[0]}: expected the work box ({x1 - x0}x{y1 - y0}) "
                     f"or the scene ({scene_hw[1]}x{scene_hw[0]})")


def _layer(image, box, scene_hw, name):
    return _box_view(_frame(image, name), box, scene_hw, name).astype(np.float64)


def _mask(mask, box, scene_hw, name):
    m = mask.detach().to("cpu", torch.float32)
    if m.dim() == 4 and m.shape[1] == 1:
        m = m[:, 0]
    if m.dim() == 3:
        if m.shape[0] != 1:
            raise ValueError(f"{name}: Image Post works on one mask per job, got a batch of {m.shape[0]}")
        m = m[0]
    if m.dim() != 2:
        raise ValueError(f"{name}: expected a MASK [B, H, W], got shape {tuple(mask.shape)}")
    return _box_view(m.numpy(), box, scene_hw, name).astype(np.float64)


def _to_layer(lin):
    """Linear float64 -> sRGB layer, values above 1 kept (imgpost's lin_to_srgb only clips below 0)."""
    from .imgpost import images as im
    return im.lin_to_srgb(lin)


def _from_layer(srgb):
    """Inverse of _to_layer. imgpost's srgb_to_lin clips at 1, which would cut the highlight headroom."""
    s = np.clip(srgb, 0, None)
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def _image_out(arr):
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None]


def _mask_out(arr):
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None]


def _pil_out(im):
    return torch.from_numpy(np.asarray(im.convert("RGB")).astype(np.float32) / 255.0)[None]


def _placeholder():
    """Stands in for a QA image the run didn't make (no edges to show, no relight)."""
    return torch.full((1, 64, 64, 3), 0.1)


def _output_file(prefix, ext):
    """-> (absolute path, filename, subfolder) in ComfyUI's output folder, numbered like Save Image's files."""
    import folder_paths   # ComfyUI's module (tools/offline_run.py provides a stand-in)
    root = os.path.abspath(folder_paths.get_output_directory())
    sub, _, stem = prefix.strip().replace("\\", "/").rpartition("/")
    stem = stem or "image-post"
    folder = os.path.abspath(os.path.join(root, sub))
    if os.path.commonpath([root, folder]) != root:
        raise ValueError(f"filename_prefix {prefix!r} points outside ComfyUI's output folder")
    os.makedirs(folder, exist_ok=True)
    pat = re.compile(re.escape(stem) + r"_(\d{5})_\." + re.escape(ext))
    taken = [int(m.group(1)) for f in os.listdir(folder) if (m := pat.fullmatch(f))]
    name = f"{stem}_{max(taken, default=0) + 1:05}_.{ext}"
    return os.path.join(folder, name), name, sub


def _scene(job, scene, name="scene"):
    """-> (scene uint8, work box [x0, y0, x1, y1], (H, W))"""
    u8 = _u8(scene, name)
    Hs, Ws = u8.shape[:2]
    if job.get("scene_size") and list(job["scene_size"]) != [Ws, Hs]:
        w, h = job["scene_size"]
        raise ValueError(f"the job was measured on a {w}x{h} scene but `{name}` is {Ws}x{Hs}: "
                         "connect the image the job was measured on")
    return u8, _stages().work_box(job, Hs, Ws), (Hs, Ws)


def _reference(job, reference, reference_mask=None):
    """-> (reference uint8, alpha uint8 or None). reference_mask is LoadImage's MASK (1 = transparent)."""
    u8 = _u8(reference, "reference")
    if job.get("reference_size") and list(job["reference_size"]) != [u8.shape[1], u8.shape[0]]:
        w, h = job["reference_size"]
        raise ValueError(f"the job was measured on a {w}x{h} reference but `reference` is {u8.shape[1]}x{u8.shape[0]}")
    alpha = None
    if reference_mask is not None:
        m = reference_mask.detach().to("cpu", torch.float32)
        m = m[0] if m.dim() == 3 else m
        if tuple(m.shape) == u8.shape[:2]:   # LoadImage returns a 64x64 placeholder when the file has no alpha
            alpha = np.clip(np.round((1.0 - m.numpy().astype(np.float64)) * 255), 0, 255).astype(np.uint8)
    return u8, alpha


def _refmask(job, ref_u8, ref_a):
    try:
        return _stages().reference_mask(job, ref_u8, ref_a)
    except ValueError as e:
        if job["reference_outline"].get("type") == "alpha":
            raise ValueError(f"{e}: connect the reference LoadImage's MASK to `reference_mask`") from e
        raise


def _box_masks(job, box, hw, product_mask, old_silhouette, occluder_mask):
    a_new = _mask(product_mask, box, hw, "product_mask")
    old = _mask(old_silhouette, box, hw, "old_silhouette") > 0.5
    vis = 1.0 - _mask(occluder_mask, box, hw, "occluder_mask")
    return a_new, old, vis


# ---- nodes ----

class ImagePostLoadJob:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_json": ("STRING", {"default": "", "multiline": True,
                                        "tooltip": "job.json text in the image-post skill's format. Ignored when job_path is set."}),
                "job_path": ("STRING", {"default": "",
                                        "tooltip": "Path to a job.json on the machine running ComfyUI. Overrides job_json."}),
            },
        }

    RETURN_TYPES = (JOB,)
    RETURN_NAMES = ("job",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Loads an image-post job (all measurements: work box, landmarks, edge lines, outlines, occluders, "
        "grade/finish/fill settings). The job's scene, reference and output paths are not read: images come "
        "from Load Image nodes and the result goes to a Save Image node."
    )

    @classmethod
    def IS_CHANGED(cls, job_json, job_path):
        if job_path.strip():
            try:
                with open(os.path.expanduser(job_path.strip()), "rb") as f:
                    return hashlib.sha256(f.read()).hexdigest()
            except OSError:
                return float("nan")
        return job_json

    def load(self, job_json, job_path):
        source = None
        if job_path.strip():
            source = os.path.abspath(os.path.expanduser(job_path.strip()))
            with open(source) as f:
                text = f.read()
        elif job_json.strip():
            text = job_json
        else:
            raise ValueError("give the job: paste job.json into job_json or set job_path")
        try:
            job = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"the job is not valid JSON: {e}") from e
        _stages().check_job(job)
        if source:
            job["_source"] = source
        return (job,)


class ImagePostJobMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job": (JOB,), "scene": ("IMAGE",)}}

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("old_silhouette", "occluder_mask")
    FUNCTION = "masks"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Rasterises the job's old_silhouette (the generated product's outline) and occluders (objects in front, "
        "anti-aliased) over the work box. Any scene- or work-box-sized MASK can stand in for either output."
    )

    def masks(self, job, scene):
        _, box, _ = _scene(job, scene)
        old, vis = _stages().scene_masks(job, box)
        return (_mask_out(old), _mask_out(1.0 - vis))


class ImagePostFitWarp:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job": (JOB,)}}

    RETURN_TYPES = (HOMOGRAPHY, "STRING")
    RETURN_NAMES = ("homography", "report")
    FUNCTION = "fit"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Fits the reference -> scene mapping (homography, affine, or a cylinder for labels on bottles, plus the "
        "optional residual edge correction) to the job's align points, y_only/x_only rows, edge lines and curves "
        "and limbs. Shows the residual table; accept visible edge lines at 1.5 px or less."
    )

    def fit(self, job):
        S = _stages()
        H, rep = S.fit_warp(job)
        return {"ui": {"text": [S.align_table(rep)]}, "result": (H, json.dumps({"align": rep}))}


class ImagePostWarpReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"job": (JOB,), "homography": (HOMOGRAPHY,), "scene": ("IMAGE",), "reference": ("IMAGE",)},
            "optional": {"reference_mask": ("MASK", {"tooltip": "The reference Load Image's MASK, for reference_outline type 'alpha'"})},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", COORDS)
    RETURN_NAMES = ("warped", "product_mask", "align_overlay", "coords")
    FUNCTION = "warp"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Warps the reference into the work box (supersampled, in linear light) and its real outline into the "
        "product mask. align_overlay: reference at 50% over the scene, real outline in green, occluders in red. "
        "coords: where each work-box pixel samples the reference (for grade.shading). Warns when the measured "
        "outline takes in the packshot's backdrop (a light rim in the composite)."
    )

    def warp(self, job, homography, scene, reference, reference_mask=None):
        su8, box, _ = _scene(job, scene)
        ru8, ra = _reference(job, reference, reference_mask)
        refmask = _refmask(job, ru8, ra)
        H = homography if hasattr(homography, "forward") else np.asarray(homography, float)
        W_lin, a_new, (U, V), overlay = _stages().warp(job, H, refmask, su8, ru8, box)
        _fr, warn = _stages().fringe(refmask, ru8, H)
        result = (_image_out(_to_layer(W_lin)), _mask_out(a_new), _pil_out(overlay), {"box": box, "u": U, "v": V})
        return {"ui": {"text": [warn or "outline clear of the packshot's backdrop"]}, "result": result}


class ImagePostGradeToScene:
    # New grade options: port the branch from run.py into stages.grade; the job's grade section reaches it as is.
    # A switch on the node is a combo whose first choice is "job", appended after the existing ones (UI workflows
    # store widget values by position), like `colour` and `shading`.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job": (JOB,), "scene": ("IMAGE",), "warped": ("IMAGE",), "product_mask": ("MASK",),
                "old_silhouette": ("MASK",), "occluder_mask": ("MASK",),
                "colour": (COLOUR_CHOICES, {"default": "job", "tooltip": (
                    "job: grade.colour from the job (default scene). scene: tone curves fitted to the generated "
                    "product. reference: keep the real product's colours, taking only light, white balance and "
                    "black level from the scene (grade.white_box, black_box, ref_white_box in the job).")}),
                "shading": (SHADING_CHOICES, {"default": "job", "tooltip": (
                    "job: apply grade.shading from the job when it has one (cylinders: the generated product's "
                    "residual light across the reference x, plus optional limb darkening and specular streak). "
                    "off: skip it, to compare.")}),
            },
            "optional": {
                "reference": ("IMAGE", {"tooltip": "Needed when the job sets grade.ref_white_box"}),
                "coords": (COORDS, {"tooltip": "Warp Reference's coords. Needed when the job sets grade.shading"}),
                "relight": (RELIGHT_CHOICES, {"default": "job", "tooltip": (
                    "job: apply grade.relight from the job when it has one (the source's cast shadows, falloff, "
                    "glows, and the gloss or wrap haze with sheen). off: skip it, to compare.")}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "IMAGE", INK)
    RETURN_NAMES = ("graded", "grade_preview", "report", "relight_map", "ink")
    FUNCTION = "grade"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Carries the scene's light and colour onto the warped reference: a smooth light falloff (gain "
        "white_level, luma_ratio or none), then per-channel tone curves (colour scene) or the real product's own "
        "colours between scene white and black points (colour reference), then grade.shading for cylinders and "
        "grade.relight for the light the smooth grade misses (cast shadows, glows, gloss). relight_map shows "
        "where relight changed the light (blue darker, red brighter); a dark placeholder when it didn't run. "
        "ink: for a job with matte 'ink' (print only, over the scene's own surface), connect it to Composite Behind."
    )

    def grade(self, job, scene, warped, product_mask, old_silhouette, occluder_mask, colour, shading,
              reference=None, coords=None, relight="job"):
        su8, box, hw = _scene(job, scene)
        gs = dict(job.get("grade", {}))
        if colour != "job":
            gs["colour"] = colour
        if shading == "off":
            gs.pop("shading", None)
        if relight == "off":
            gs.pop("relight", None)
        a_new, old, vis = _box_masks(job, box, hw, product_mask, old_silhouette, occluder_mask)
        ru8 = _reference(job, reference)[0] if reference is not None else None
        if coords is not None and list(coords["box"]) != list(box):
            raise ValueError(f"coords were computed for work box {coords['box']}, not {box}: re-run Warp Reference")
        graded, rep, preview, sheet, ink = _stages().grade(gs, su8, ru8, box, _from_layer(_layer(warped, box, hw, "warped")),
                                                           a_new, old, vis, None if coords is None else coords["u"],
                                                           job.get("matte"))
        return (_image_out(_to_layer(graded)), _pil_out(preview), json.dumps({"grade": rep}),
                _pil_out(sheet) if sheet else _placeholder(), ink)


class ImagePostMatchFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job": (JOB,), "scene": ("IMAGE",), "graded": ("IMAGE",), "product_mask": ("MASK",),
                             "old_silhouette": ("MASK",), "occluder_mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("product", "report")
    FUNCTION = "finish"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Rolls off highlights, then matches the frame's softness (finish.blur, auto from edge steepness in "
        "blur_probe) and grain (finish.grain, auto from MAD noise in flat areas)."
    )

    def finish(self, job, scene, graded, product_mask, old_silhouette, occluder_mask):
        su8, box, hw = _scene(job, scene)
        a_new, old, vis = _box_masks(job, box, hw, product_mask, old_silhouette, occluder_mask)
        card, rep = _stages().finish(job.get("finish", {}), job.get("grade", {}), su8, box,
                                     _from_layer(_layer(graded, box, hw, "graded")), a_new, old, vis)
        return (_image_out(card), json.dumps({"finish": rep}))


class ImagePostFillLeftovers:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job": (JOB,), "scene": ("IMAGE",), "product_mask": ("MASK",),
                             "old_silhouette": ("MASK",), "occluder_mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("background", "fill_mask", "report")
    FUNCTION = "fill"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Harmonic (membrane) fill of the background where the generated product showed and the real one doesn't, "
        "with matched grain. Stops with FILL STOPPED when a leftover region has no background around it: extend "
        "reference_outline under the occluder or correct the occluder polyline."
    )

    def fill(self, job, scene, product_mask, old_silhouette, occluder_mask):
        su8, box, hw = _scene(job, scene)
        a_new, old, vis = _box_masks(job, box, hw, product_mask, old_silhouette, occluder_mask)
        base, F, rep = _stages().fill(job.get("fill", {}), su8, box, a_new, old, vis, job.get("matte"))
        return (_image_out(base), _mask_out(F), json.dumps({"fill": rep}))


class ImagePostCompositeBehind:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job": (JOB,), "scene": ("IMAGE",), "background": ("IMAGE",), "product": ("IMAGE",),
                             "product_mask": ("MASK",), "occluder_mask": ("MASK",)},
                "optional": {"ink": (INK, {"tooltip": "Grade To Scene's ink: needed when the job sets matte 'ink'"})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "matte")
    FUNCTION = "composite"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Composites the product over the filled background behind the occluders (matte = product mask softened by "
        "finish.edge_softness, times visibility) and pastes the work box into the scene, quantised to 8 bit like "
        "the skill's output. Pixels outside the work box are passed through untouched."
    )

    def composite(self, job, scene, background, product, product_mask, occluder_mask, ink=None):
        _, box, hw = _scene(job, scene)
        out, alpha = _stages().composite(job.get("finish", {}), _layer(background, box, hw, "background"),
                                         _layer(product, box, hw, "product"),
                                         _mask(product_mask, box, hw, "product_mask"),
                                         1.0 - _mask(occluder_mask, box, hw, "occluder_mask"), ink, job.get("matte"))
        x0, y0, x1, y1 = box
        image = scene[:1].detach().to("cpu", torch.float32).clone()
        image[0, y0:y1, x0:x1, :3] = torch.from_numpy(out.astype(np.float32) / 255.0)
        return (image, _mask_out(alpha))


class ImagePostQASheet:
    @classmethod
    def INPUT_TYPES(cls):
        report = ("STRING", {"forceInput": True})
        return {
            "required": {"job": (JOB,), "before": ("IMAGE",), "after": ("IMAGE",), "reference": ("IMAGE",),
                         "matte": ("MASK",), "fill_mask": ("MASK",)},
            "optional": {"reference_mask": ("MASK",), "align_report": report, "grade_report": report,
                         "finish_report": report, "fill_report": report},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("compare", "edges", "before_after", "report")
    FUNCTION = "qa"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "QA sheets: compare (before, after, reference), edges (4x before|after tiles along every boundary and "
        "fill) and before_after, plus the run's report.json (stage reports merged, change counts including "
        "outside_roi_changed). Look at the sheets before delivering."
    )

    def qa(self, job, before, after, reference, matte, fill_mask, reference_mask=None,
           align_report=None, grade_report=None, finish_report=None, fill_report=None):
        S = _stages()
        su8, box, hw = _scene(job, before, "before")
        res = _u8(after, "after")
        if res.shape != su8.shape:
            raise ValueError(f"`after` is {res.shape[1]}x{res.shape[0]} but `before` is {su8.shape[1]}x{su8.shape[0]}")
        ru8, ra = _reference(job, reference, reference_mask)
        rep = {"job": job["_source"]} if job.get("_source") else {}
        rep.update({"roi": box, "stage": "all"})
        for part in (align_report, grade_report, finish_report, fill_report):
            if part:
                rep.update(json.loads(part))
        rep["change"] = S.change_report(res, su8, box)
        before_after, compare, tiles = S.qa_sheets(res, su8, ru8, _refmask(job, ru8, ra),
                                                   _mask(matte, box, hw, "matte"),
                                                   _mask(fill_mask, box, hw, "fill_mask") > 0.5, box)
        edges = _pil_out(tiles) if tiles else _placeholder()
        text = json.dumps(rep, indent=2)
        return {"ui": {"text": [text]}, "result": (_pil_out(compare), edges, _pil_out(before_after), text)}


class ImagePostSavePSD:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "job": (JOB,), "scene": ("IMAGE",), "image": ("IMAGE",), "background": ("IMAGE",), "fill_mask": ("MASK",),
            "product": ("IMAGE",), "matte": ("MASK",),
            "filename_prefix": ("STRING", {"default": "image-post/refined", "tooltip": (
                "Folder and name inside ComfyUI's output folder; files are numbered like Save Image's.")}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Writes the result as a layered PSD for finishing by hand (run.py --psd): the original scene, the "
        "background fill (opaque only where the fill ran) and the product with matte x occluders as its layer "
        "mask. With every layer visible it flattens to the PNG within 2 levels (report: layers_vs_png). "
        "scene: the frame before this job; image: Composite Behind's image."
    )

    def save(self, job, scene, image, background, fill_mask, product, matte, filename_prefix):
        su8, box, hw = _scene(job, scene)
        res = _u8(image, "image")
        if res.shape != su8.shape:
            raise ValueError(f"`image` is {res.shape[1]}x{res.shape[0]} but `scene` is {su8.shape[1]}x{su8.shape[0]}")
        path, name, sub = _output_file(filename_prefix, "psd")
        rep = _stages().write_psd(path, su8, res, _layer(background, box, hw, "background"),
                                  _mask(fill_mask, box, hw, "fill_mask") > 0.5, _layer(product, box, hw, "product"),
                                  _mask(matte, box, hw, "matte"), box)
        rep["path"] = f"{sub}/{name}" if sub else name   # inside ComfyUI's output folder
        text = json.dumps({"psd": rep}, indent=2)
        return {"ui": {"text": [text], "files": [{"filename": name, "subfolder": sub, "type": "output"}]}}


NODE_CLASS_MAPPINGS = {
    "ImagePostLoadJob": ImagePostLoadJob,
    "ImagePostJobMasks": ImagePostJobMasks,
    "ImagePostFitWarp": ImagePostFitWarp,
    "ImagePostWarpReference": ImagePostWarpReference,
    "ImagePostGradeToScene": ImagePostGradeToScene,
    "ImagePostMatchFinish": ImagePostMatchFinish,
    "ImagePostFillLeftovers": ImagePostFillLeftovers,
    "ImagePostCompositeBehind": ImagePostCompositeBehind,
    "ImagePostQASheet": ImagePostQASheet,
    "ImagePostSavePSD": ImagePostSavePSD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePostLoadJob": "Image Post: Load Job",
    "ImagePostJobMasks": "Image Post: Job Masks",
    "ImagePostFitWarp": "Image Post: Fit Warp",
    "ImagePostWarpReference": "Image Post: Warp Reference",
    "ImagePostGradeToScene": "Image Post: Grade To Scene",
    "ImagePostMatchFinish": "Image Post: Match Finish",
    "ImagePostFillLeftovers": "Image Post: Fill Leftovers",
    "ImagePostCompositeBehind": "Image Post: Composite Behind",
    "ImagePostQASheet": "Image Post: QA Sheet",
    "ImagePostSavePSD": "Image Post: Save PSD",
}
