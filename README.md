# CodeFormer API — Python client

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Hosted on Synexa](https://img.shields.io/badge/hosted%20on-Synexa-6366f1.svg)](https://synexa.ai/explore/sczhou/codeformer?utm_source=github&utm_medium=ugc&utm_campaign=codeformer-dev&utm_content=readme-badge&utm_term=tier-a)

CodeFormer is the NeurIPS 2022 face restoration model from NTU's S-Lab that rebuilds degraded faces by looking up a learned codebook of high-quality facial features, with a single knob to trade fidelity against quality. This package is a Python client for the CodeFormer API hosted on Synexa: one `pip install` and a `run({"image": url})` call return a restored image, with GFPGAN available on the same client as a lighter alternative.

You get a blocking `run()` that waits for the result, a submit-and-poll mode for batches, webhook delivery on completion and typed errors. The only dependency is `httpx`. It is meant for photo-restoration products, avatar pipelines, archival digitisation and anyone post-processing AI-generated portraits who does not want to maintain a CUDA environment for a two-second job.

> **Try it now:** [https://synexa.ai/explore/sczhou/codeformer](https://synexa.ai/explore/sczhou/codeformer?utm_source=github&utm_medium=ugc&utm_campaign=codeformer-dev&utm_content=readme-top&utm_term=tier-a) — the hosted model behind this client. New accounts get a free trial credit.

## Contents

- [Why this client](#why-this-client)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Hosted models](#hosted-models)
- [Parameters](#parameters)
- [Advanced usage](#advanced-usage)
- [About CodeFormer](#about-codeformer)
- [Use cases](#use-cases)
- [FAQ](#faq)
- [License](#license)

## Why this client

- **The dependency stack is the hard part, not the model.** CodeFormer itself is small, but a working install needs PyTorch with CUDA, facexlib for face detection and alignment, Real-ESRGAN for the background, and pretrained weights for each. The hosted endpoint has all of that assembled; you call HTTPS.
- **No GPU on the calling side.** Restoring a face on CPU is too slow for a request path. With the hosted model your web server, notebook or serverless function stays lightweight.
- **No cold start per request.** Detector, restorer and upsampler are already loaded on the hosted instance, so a single image is answered at inference speed.
- **$0.0025 per image ($0.0008 with GFPGAN).** Billing is per run with nothing charged while idle; ten thousand restored photos cost $25.

## Installation

```bash
pip install git+https://github.com/codeformer-dev/codeformer-api.git
```

Then set your API key (create one at [synexa.ai](https://synexa.ai?utm_source=github&utm_medium=ugc&utm_campaign=codeformer-dev&utm_content=readme-apikey&utm_term=tier-a)):

```bash
export SYNEXA_API_KEY="sk-..."
```

## Quickstart

```python
import codeformer_api

output = codeformer_api.run({
    "image": "https://example.com/input.png"
})
print(output)   # URL(s) of the generated result
```

Or with an explicit client:

```python
from codeformer_api import Client

client = Client(api_key="sk-...")
output = client.run({"image": "https://example.com/input.png"})
```

## Hosted models

| Model | Category | What it does | Price / run |
|---|---|---|---|
| [`sczhou/codeformer`](https://synexa.ai/explore/sczhou/codeformer?utm_source=github&utm_medium=ugc&utm_campaign=codeformer-dev&utm_content=readme-models&utm_term=tier-a) | super-resolution | Robust face restoration algorithm for old photos / AI-generated faces | $0.0025 |
| [`tencentarc/gfpgan`](https://synexa.ai/explore/tencentarc/gfpgan?utm_source=github&utm_medium=ugc&utm_campaign=codeformer-dev&utm_content=readme-models&utm_term=tier-a) | super-resolution | Practical face restoration algorithm for *old photos* or *AI-generated faces* | $0.0008 |

The default model is **`sczhou/codeformer`**; pass `model="owner/name"` to `run()` to use another one from the table.

## Parameters

### `sczhou/codeformer`

| Field | Type | Required | Default | Range | Description |
|---|---|---|---|---|---|
| `image` | file | yes | `https://synexa.s3.us-east-005.backblazeb…` | — | Input image |
| `upscale` | integer | no | `2` | 1, 16 | The final upsampling scale of the image |
| `face_upsample` | boolean | no | `True` | — | Upsample restored faces for high-resolution AI-created images |
| `background_enhance` | boolean | no | `True` | — | Enhance background image with Real-ESRGAN |
| `codeformer_fidelity` | number | no | `0.5` | 0, 1 | Balance the quality (lower number) and fidelity (higher number). |

### `tencentarc/gfpgan`

| Field | Type | Required | Default | Range | Description |
|---|---|---|---|---|---|
| `img` | file | yes | `https://synexa.s3.us-east-005.backblazeb…` | — | Input |
| `scale` | number | no | `2` | 1, 16 | Rescaling factor |
| `version` | string | no | `v1.4` | v1.2, v1.3, v1.4 | GFPGAN version. v1.3: better quality. v1.4: more details and better identity. |

## Advanced usage

**Submit without blocking, then poll:**

```python
prediction = client.run(input, wait=False)      # returns immediately
prediction = client.wait(prediction, timeout=300)
print(prediction["output"])
```

**Webhook on completion:**

```python
client.run(input, wait=False, webhook="https://your-app.example/hooks/synexa")
```

**Errors:**

```python
from codeformer_api import ModelError, PredictionTimeout

try:
    output = client.run(input)
except ModelError as e:
    print("failed:", e, e.prediction and e.prediction.get("id"))
except PredictionTimeout:
    print("still running — poll later")
```

Status values you will see on a prediction: `starting` → `processing` → `succeeded` | `failed`.

## About CodeFormer

CodeFormer is a blind face restoration model introduced in the NeurIPS 2022 paper *Towards Robust Blind Face Restoration with Codebook Lookup Transformer* by Shangchen Zhou, Kelvin Chan, Chongyi Li and Chen Change Loy of S-Lab, Nanyang Technological University. Code and weights are published at [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) under the S-Lab License (non-commercial); check its terms for your use. "Blind" means the model receives no information about how the input was degraded: blur, noise, JPEG artefacts, low resolution and old-print damage are all handled by the same network.

The method has two stages. A VQGAN-style autoencoder first learns a discrete codebook of high-quality facial features from clean faces; a transformer is then trained to predict the right code sequence from a degraded input, which turns restoration into classification over the codebook and keeps the output robust even when most input detail is gone. A controllable feature transformation module blends the degraded input back in, governed by a fidelity weight `w` between 0 and 1: low values give cleaner, more idealised faces, high values stay closer to the original identity at the cost of sharpness. The full pipeline detects and aligns faces, restores each and pastes them back, optionally upsampling the background with Real-ESRGAN.

The hosted `sczhou/codeformer` endpoint exposes that pipeline: `image` is the only required input, `codeformer_fidelity` is the `w` knob, and `upscale`, `background_enhance` and `face_upsample` control the output scale and the optional enhancers. The client also exposes `tencentarc/gfpgan`, Tencent ARC's GFPGAN (CVPR 2021), which restores faces with a StyleGAN2 generative prior; it is cheaper and faster, though it tends to smooth heavily degraded inputs more than CodeFormer does. Both work on single images; neither is a video tool.

The endpoints used by this client are `sczhou/codeformer` and `tencentarc/gfpgan`, which are the same CodeFormer and GFPGAN models released by their authors, served on Synexa. The weights and reference inference scripts are in the official repositories if you would rather self-host.

**Official project:** https://github.com/sczhou/CodeFormer

## Use cases

- **Old family photo restoration** — call `run({"image": scan_url, "codeformer_fidelity": 0.7, "background_enhance": True})` to sharpen faces while keeping likeness high and clean up the surrounding print.
- **Fixing AI-generated portraits** — post-process text-to-image outputs with `face_upsample=True` and a low fidelity value to correct malformed eyes and skin texture.
- **Profile-picture enhancement in a consumer app** — accept a user upload, restore it at `upscale=2`, and return the URL within a single request.
- **Archival digitisation at scale** — submit a museum or newspaper collection with `wait=False` and a `webhook` and store restored versions alongside the originals.
- **Video-call or ID-photo pre-processing** — run low-light webcam captures through GFPGAN (`model="tencentarc/gfpgan"`) where speed matters more than maximum detail.
- **Comparing restorers** — send the same image to both endpoints from one client and let users pick, since CodeFormer and GFPGAN differ in how much they idealise a face.

## FAQ

**Is there a CodeFormer API?**

The authors publish CodeFormer as code and weights, not as a hosted service. This package is a Python client for the `sczhou/codeformer` endpoint on Synexa, which serves the released model behind an HTTPS API, alongside `tencentarc/gfpgan`.

**How much does the CodeFormer API cost?**

`sczhou/codeformer` is $0.0025 per run and `tencentarc/gfpgan` is $0.0008 per run. Billing is per prediction with no idle charge; new Synexa accounts receive a free trial credit.

**Can I run CodeFormer without a GPU?**

Yes. With this client the restoration runs on Synexa's GPUs and you receive an image URL; your environment needs only Python 3.8+ and `httpx`. Self-hosting needs a CUDA GPU plus PyTorch, facexlib and Real-ESRGAN weights.

**Does this client work with the sczhou/CodeFormer repo or ComfyUI?**

No. It does not load local weights or ComfyUI nodes; it is an HTTP client for the hosted endpoint. Use the official repository or its ComfyUI ports for offline inference, video frames or custom fidelity schedules.

**What input formats does it accept?**

`image` (CodeFormer) or `img` (GFPGAN) must be a publicly reachable image URL, typically JPEG or PNG. CodeFormer's optional fields are `codeformer_fidelity` (0–1), `upscale` (integer), `background_enhance` and `face_upsample` (booleans). GFPGAN takes `scale` (number) and `version` (`v1.3` or `v1.4`). Output is a URL to the restored image.

**Is this the official CodeFormer SDK?**

No. This is an independent, MIT-licensed client and is not affiliated with S-Lab, NTU or the CodeFormer authors. The official project is at https://github.com/sczhou/CodeFormer.

## Related

- [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) — official code, weights and paper for CodeFormer.
- [Synexa Python client](https://github.com/synexa-ai/synexa-python) — the general-purpose client for every model on the platform.
- [tencentarc/gfpgan](https://synexa.ai/explore/tencentarc/gfpgan) — GFPGAN face restoration, also supported by this client.
- [bytedance/seedvr2-upscale](https://synexa.ai/explore/bytedance/seedvr2-upscale) — whole-image diffusion upscaling to pair with face restoration.
- [black-forest-labs/flux-kontext-pro](https://synexa.ai/explore/black-forest-labs/flux-kontext-pro) — prompt-based edits once the face is restored.

## License

MIT. This is an independent, community-maintained client and is not affiliated with or endorsed by the authors of CodeFormer. Model weights and trademarks belong to their respective owners.

_Last reviewed: 2026-09-22_
