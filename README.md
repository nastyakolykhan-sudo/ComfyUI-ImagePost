# ComfyUI-ImagePost

Fix a hallucinated product in an AI-generated still by compositing the real product from its packshot, inside ComfyUI. The nodes warp the packshot onto the generated product's perspective and carry over the scene's light and colour. That includes cast shadows, falloff, gloss and haze, and the round light on cans and bottles. They match the frame's softness and grain, put the product behind whatever stands in front of it, and fill in the background where the generated product was bigger. Every run ends with QA sheets, a report, and a layered PSD if you want to finish by hand.

It is deterministic compositing with no models. Every measurement lives in a job file (`job.json`): the work box, landmark points, edge lines, outlines, foreground objects, and the grade, finish and fill settings. The same job gives the same pixels every time. That lets an agent or a script drive it through ComfyUI's API, check the report and QA sheets, edit the job and run it again.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nastyakolykhan-sudo/ComfyUI-ImagePost
```

It needs no extra dependencies: numpy, scipy and Pillow (10.1 or later) ship with ComfyUI. Restart ComfyUI. The nodes are under **image › image-post**.

## Try the demo

1. Copy `examples/demo/imagepost_demo_scene.png` and `examples/demo/imagepost_demo_packshot.png` into `ComfyUI/input/`.
2. Load `workflows/image-post.json` and queue it.
3. The refined image lands in `output/image-post/`, the QA sheets in `output/image-post/qa/`, and the layered PSD next to the image.

The demo is a made-up product card. In the "generated" scene its text is garbled ("PRDOUCT", "sapmle lable"), its emblem and grid are wrong, its corners are square and its top is too tall. A mug stands in front of its lower right corner and casts a shadow across it. The packshot is the clean card. `tools/make_demo.py` draws all three files from code, so they show no real product or brand. `examples/demo/job.json` is the worked example of a job.

## Nodes

There is one node per step. The math is the image-post engine's (`imgpost/`, see [Maintaining](#maintaining)), split at the node boundaries in `stages.py`.

| Node | Inputs | Outputs | Step |
|---|---|---|---|
| **Load Job** | `job_json` (text) or `job_path` | `job` | reads and checks the job |
| **Job Masks** | job, scene | `old_silhouette`, `occluder_mask` | rasterises the generated outline and the foreground objects |
| **Fit Warp** | job | `homography`, `report` | homography, affine or cylinder fit (labels on bottles), with the optional residual edge correction; shows the residual table |
| **Warp Reference** | job, homography, scene, reference, `reference_mask`? | `warped`, `product_mask`, `align_overlay`, `coords` | supersampled warp in linear light; warns when the outline takes in the packshot's backdrop (a light rim) |
| **Grade To Scene** | job, scene, warped, masks, `colour`, `shading`, `reference`?, `coords`?, `relight`? | `graded`, `grade_preview`, `report`, `relight_map`, `ink` | light falloff, colour, the scene's paper field, cylinder shading, relight; `ink` feeds Composite Behind for jobs with `matte: ink` |
| **Match Finish** | job, scene, graded, masks | `product`, `report` | highlight roll-off, blur and grain match |
| **Fill Leftovers** | job, scene, masks | `background`, `fill_mask`, `report` | harmonic fill; stops with FILL STOPPED |
| **Composite Behind** | job, scene, background, product, product_mask, occluder_mask, `ink`? | `image`, `matte` | composite and paste into the scene; with `matte: ink`, only the print goes over the scene's own surface |
| **QA Sheet** | job, before, after, reference, matte, fill_mask, reports? | `compare`, `edges`, `before_after`, `report` | QA sheets and `report.json` |
| **Save PSD** | job, scene, image, background, fill_mask, product, matte, `filename_prefix` | (file) | layered PSD for hand finishing |

`?` marks optional inputs. The masks inputs are `product_mask`, `old_silhouette` and `occluder_mask`.

A switch set to `job` does what the job says, so the switches only ever override it:

- `colour`: `job` uses `grade.colour` (default `scene`). `scene` fits tone curves to the generated product. `reference` keeps the real product's colours and takes only light, white balance and black level from the scene (`grade.white_box`, `black_box`, `ref_white_box`). Use `reference` whenever the generated colours are off, as in the demo. It needs `reference` connected when the job has `ref_white_box`.
- `shading`: `job` applies `grade.shading` for cans and bottles: the generated product's round light, highlight streaks and core shadow. It needs `coords` from Warp Reference. `off` skips it, for comparison.
- `relight`: `job` applies `grade.relight`, which brings back the light the smooth grade misses: cast shadows with their crisp edges, falloff and glows, plus gloss or wrap haze with `sheen`. `relight_map` shows what it changed (blue darker, red brighter, sheen in white). `off` skips it, for comparison.

Save PSD writes three layers, bottom to top:
- `original scene`;
- `background fill`, opaque only where the fill ran;
- `product`, with matte × foreground visibility as its layer mask.

With every layer visible it flattens to the PNG within 2 levels (its report: `layers_vs_png`). In the editor it runs on every queue, so mute it (Ctrl+M) when you don't need a PSD. `tools/comfy_run.py` runs it only with `--psd`.

## Conventions

- **Work box.** `scene` is the whole generated frame, and the job's `roi` is the work box. Every other IMAGE or MASK passed between Image Post nodes covers only the work box. Composite Behind pastes it back, and every pixel outside the box passes through untouched.
- **Masks.** Mask inputs also accept a scene-sized mask, which they crop to the work box, so a mask from the mask editor or another node can stand in. `old_silhouette` is thresholded at 0.5. `occluder_mask` is 1 where something is in front of the product.
- **Product layers.** `warped`, `graded` and `product` are sRGB like any IMAGE, but values above 1.0 are kept as highlight headroom, and Match Finish rolls them off. Don't clamp them in between.
- **One image per job.** Batches larger than 1 raise an error.
- **Images come from Load Image nodes.** The job's `scene`, `reference` and `output` paths aren't read here. When the job has `scene_size` or `reference_size`, the connected images must match, which catches a wrong wire. A transparent packshot (`reference_outline` type `alpha`) needs the reference Load Image's MASK connected to `reference_mask`.
- **Output.** Composite Behind quantises the work box to 8 bit, so Save Image writes the same pixels the engine saves. Save Image and Save PSD don't keep the scene's ICC profile or alpha channel; the engine's own runner does.
- **Several products in one frame.** Run one job per product, back to front, and feed each run's image in as the next run's scene.

## Running it from an agent (ComfyUI API)

`tools/comfy_run.py` uses only the standard library. It uploads the images, swaps in a job, queues the prompt, waits, then downloads everything:

```bash
python3 tools/comfy_run.py --server http://127.0.0.1:8188 \
  --workflow workflows/image-post.api.json \
  --scene examples/demo/imagepost_demo_scene.png --reference examples/demo/imagepost_demo_packshot.png \
  --job examples/demo/job.json --out ./ip-run
```

- `--scene` and `--reference` upload to `input/image-post/` and set the Load Image nodes that feed Warp Reference. `--image NODE_ID=PATH` targets any other Load Image node.
- `--job` replaces the job in every Load Job node. Without it, the workflow's embedded job runs.
- `--stage align` runs only the fit, warp and `align_overlay`. `--stage grade` adds `grade_preview` and `relight_map`. Use them to check the alignment, then the grade, before a full run.
- `--psd` also runs Save PSD, as does `"psd": true` in the job.
- `--out` gets the outputs in the server's folder layout: images, QA sheets and the PSD. It also gets `report.json` (the QA node's report), `ImagePostFitWarp-5.txt` (the residual table), `ImagePostSavePSD-19.txt`, `prompt.json` and `history.json`.
- Exit codes: 0 done, 2 prompt rejected (validation errors printed), 3 node failed (for example `FILL STOPPED ...`, printed with the node), 4 timeout.
- For a server behind an auth proxy, pass `--header "Name: value"` (repeatable).

The same steps with curl. Upload both images:

```bash
curl -s -F "image=@scene.png" -F "subfolder=image-post" -F "overwrite=true" http://127.0.0.1:8188/upload/image
```

Point nodes 1 (scene) and 2 (reference) at `image-post/<name>`, then queue the prompt:

```bash
jq '{prompt: (.["1"].inputs.image = "image-post/scene.png" | .["2"].inputs.image = "image-post/packshot.png")}' \
  workflows/image-post.api.json > prompt.json
curl -s -H "Content-Type: application/json" -d @prompt.json http://127.0.0.1:8188/prompt
```

This returns `{"prompt_id": "..."}`. Poll until the entry appears, then read `outputs`:

```bash
curl -s http://127.0.0.1:8188/history/<prompt_id>
```

In `outputs`:
- each Save Image node lists `images`;
- the QA node lists `text`, which holds `report.json`;
- Save PSD lists `files`.

Fetch each file:

```bash
curl -s -o refined.png "http://127.0.0.1:8188/view?filename=imagepost_demo_scene_refined_00001_.png&subfolder=image-post&type=output"
```

A failed run has `status.status_str: "error"`, with the node and `exception_message` under `status.messages`.

## Writing a job

`examples/demo/job.json` shows every section. The main ones:

- `roi`: the work box in scene px. The product plus about 40 px of background.
- `reference_outline`: the real product's outline in packshot px. Types: `rounded_rect`, `profile`, `polygon`, `key_white`, `alpha`. `extend` lets an edge run on behind a foreground object.
- `align`:
  - `points`: 3 or more landmarks read on both images, such as wordmark or panel corners.
  - `lines`: visible outline edges.
  - `y_only` / `x_only`: rows or columns of repeated items.
  - `model`: `homography`, `affine`, or `cylinder` for print on a bottle or jar (`cylinder`, `limbs`, `"curve": true` lines).
  - `residual`: a smooth correction onto the generated edges a rigid fit can't follow.
- `old_silhouette`: the generated product's outline in scene px. The fill restores the background inside it wherever the real product doesn't reach.
- `occluders`: objects in front, each one of:
  - a `polyline` plus a `side`;
  - a `polygon`;
  - either of those with `soften` for an out-of-focus edge.
- `grade`: `gain` (`white_level`, `luma_ratio`, `none`), `colour`, `shading`, `relight`. On printed packaging whose generated print is laid out differently, `luma_ratio` with `match_hue` reads the light only where both images show the same ink; `protect_white` keeps the packshot's clipped whites from being dimmed. For labels, `ref_flatten` takes the packshot's own light off its paper and `white_field` lays the generated label's paper colour (light, falloff, tint) under it.
- `finish`: `blur` and `grain` (`auto` or a value), `edge_softness`, `rolloff`, `sharpen: auto` (a packshot softer than the frame).
- `fill`: `dilate`, `orphan_px`.
- `prefilter`: `auto` for a small packshot enlarged in the frame. `matte`: `ink` for text printed on a coloured body.
- `psd`: true to also write the layered PSD.

Measure landmarks and edges by colour change, not by the strongest gradient, and accept a fit at 1.5 px or less on the visible edge lines. The image-post engine's job spec documents every field.

## Regression

`tools/regress.py` runs the engine's own runner (`scripts/run.py`) and this pack on the same job and images, then compares them. The runner writes into a scratch folder, never to the job's own output. The check passes when fewer than 20 pixels differ by more than 8 levels and nothing outside the work box differs. With `--psd`, both sides also write the PSD, and their layers must match and flatten within 2 levels.

```bash
python -I tools/regress.py --psd                                    # the demo, in-process; needs a Python with torch
python3 tools/regress.py --server http://127.0.0.1:8188             # the pack on a live ComfyUI server
python -I tools/regress.py --job path/to/job.json [--scene ...] [--reference ...] [--psd]
```

In-process results on 2026-10-08:

| Run | Result |
|---|---|
| The demo, with `--psd` | identical pixels; the PSDs are byte-identical |
| 17 production jobs from one 4096 × 4096 frame: cards, boxes and a pouch, glossy and plastic-wrapped packs with relight and sheen, six cans with cylinder shading, a partly hidden pack fitted to its print's perspective, a user-supplied fixed render | all pass. 8 jobs identical, 8 within 1 level on 1 or 2 pixels, 1 with a 12 × 11 px patch up to 8 levels. The 3 PSDs checked have the engine's layers and flatten within 1 level |
| The same 17 jobs chained back to front through the pack alone, against the frame the engine delivered | within 1 level: 734 of 16.8 million pixels differ |
| Labels on bottles (engine 0.5.0): cylinder fits with and without the residual correction at 4096 and 1024 px, a flattened packshot under the scene's paper field, auto sharpen, and a text block with the ink matte | all pass, within 1 level on 0 to 10 pixels |

ComfyUI passes images between nodes in 32-bit floats, where the engine keeps 64-bit, so a value can land on the other side of a threshold. The 12 × 11 px patch is where relight's fine pass kept a slightly different region. Nothing outside the work boxes differed in any run.

`tools/offline_run.py` runs any API workflow in-process, with no server. It uses ComfyUI's own LoadImage and SaveImage conversions and validates inputs and link types the way ComfyUI does. `regress.py` uses it, and it takes the same arguments as `comfy_run.py`.

## Maintaining

`imgpost/` is a byte-identical copy of the image-post engine's library. The pack vendors it rather than importing it so the math stays identical and the pack stays self-contained on any server. `imgpost.lock.json` records the sha256 of every vendored file and of the runner version that `stages.py` ports.

```bash
python3 tools/sync_imgpost.py                    # check: local edits, engine changes, runner changes
python3 tools/sync_imgpost.py --update           # copy the engine's imgpost/ here and re-pin
python3 tools/sync_imgpost.py --mark-ported      # after porting a runner change into stages.py
python -I tools/build_workflows.py               # rebuild workflows/ after changing a node's inputs or outputs
python3 tools/make_demo.py                       # redraw examples/demo/ (byte-identical with the same libraries)
```

Never edit `imgpost/` here. To pick up an engine change:
1. Change the engine.
2. Run `--update`.
3. Port any runner diff into `stages.py`.
4. Run `tools/regress.py`.
5. Commit.

## Extending

- **New grade options.** Port the branch from the runner into `stages.grade`; the job's `grade` section passes it through unchanged. To add a switch on the node, add an optional combo whose first choice is `job`, after the existing ones, as `relight` was. UI workflows store widget values by position.
- **Measuring helpers.** `imgpost/autofit.py` holds the engine's measuring helpers. They snap a rough polyline to the colour edge below a pixel, refine a rough landmark by correlation with a trust verdict, trace a packshot's outline, and propose edge lines and the old silhouette from a rough placement. The engine's own `draft` command uses them, and an Auto Draft node could wrap them: a rough quad in, a draft job out. On a clean scene a draft is close to final; on a cluttered shelf it is a scaffold to finish by hand. Mask inputs already take scene-sized masks, so a mask from another node can replace Job Masks' outputs. Score any automation against hand-measured jobs with `tools/regress.py --job` before trusting it.

## Not covered yet

- Batch mode: one job across many frames.
- A GPU path. Everything runs the engine's numpy and scipy code on the CPU, about 4 to 15 s per product.
