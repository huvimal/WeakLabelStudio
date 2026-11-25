# WeakLabelStudio

Learn reliability weights for noisy labeling functions and compare majority voting.

**Specific scope:** Supervised gold-fitted LF weighting on synthetic numeric alerts; not an unsupervised label model.

**A small personal engineering experiment, not a deployed product.** Read `experiment.py`, measured `reports/RESULTS.md` after running, and the explicit result scope before presenting this project.

## Reproduce

Python 3.11. Standalone installation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B run.py
Start-Process .\reports\report.html
```

CPU PyTorch packages can be installed first from https://download.pytorch.org/whl/cpu. Optional models use **local safetensors**, no remote model code. `common.py` documents model discovery locations and environment variables. The series manager shares an environment to avoid 36 duplicate environments; this repository also includes its own requirements for standalone use.

Series installation:

```powershell
Set-Location "D:\AI-Projects\outputs\AI-40-SelfServe"
.\.venv\Scripts\python.exe -B manager.py run --project 5
```

Setup profile: **core**. Additional prerequisites: none beyond profile dependencies. See the series START_HERE.md for complete setup. Never interpret missing models/tools or failed assertions as a successful experiment.

## Evidence

`run.py` runs the actual implementation and its domain assertions. A passing run writes JSON with source SHA-256 fingerprints, dependency versions, actual measured results and scope; Markdown and escaped offline HTML are derived from that JSON. There is no prefilled success report. A failed run saves `private/last-error.txt` and blocks publication-kit generation. Passed checks establish local consistency, not independent scientific validation.

## Experimental scope and data

This source is a compact initial experiment. Its exact assumptions, fixture sizes, baseline and limitations are returned in the report's `scope` and result fields. Most data is authored synthetic numerical/text/image/audio fixtures under MIT. Do not describe generated fixtures as real customer data, independently labelled benchmarks or production traffic. Seeds and split logic are visible in source. Do not tune on evaluation outputs and then describe them as untouched test results.

Encoder projects use a pretrained MiniLM base; cached/speculative decoding uses pretrained SmolLM2; speech uses pretrained Whisper. Those weights are **not bundled**, and their upstream licenses and model cards apply. Model setup records pinned revisions. No provider API is used.

For speech, supply only permitted, nonconfidential audio. The optional local Windows speech-synthesis preparation creates **synthetic speech**, which must be labelled as such. Real audio/transcripts and any errors stay under private/. OCR uses a separately installed trusted Tesseract executable; no OCR is mocked if it is missing.

## Presentation and limitations

No guaranteed accuracy, speedup, privacy, security, legal clearance or production suitability. Keep poor results and error slices. Do not compare methods with different data access or invent client impact. Some experiment names reflect the research topic; only the paths implemented and measured here can be claimed. In particular, projection-head training is not transformer-backbone fine-tuning, an authored graph is not optimized hnswlib, synthesized tones are not real environmental recordings, and the image retrieval core is not a deployed network service.

After inspecting a passing run, generate the external publication kit through the series manager. No tool commits, pushes, creates public repositories, posts to LinkedIn or changes Upwork automatically. The software does not contact external inference APIs or execute text as instructions.

## License

Repository code and authored fixtures: MIT, copyright 2026 Le Mai Vinh Hung. Third-party packages/models retain their own licenses. Do not commit model weights, private recordings, credentials or confidential datasets.
