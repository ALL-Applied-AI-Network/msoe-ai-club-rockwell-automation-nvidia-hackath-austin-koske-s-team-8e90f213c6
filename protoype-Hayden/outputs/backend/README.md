# Changeover-aware vial inspection backend

This FastAPI service is the camera-to-website bridge. A camera or edge client posts each frame and available detections to the API. The backend applies the current inspection logic, then pushes the frame, mask proposal, color alert, recipe context, and timestamps to website clients over a WebSocket.

## Run locally

```powershell
cd outputs/backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The interactive API contract is available at `http://localhost:8000/docs`. Set `FRONTEND_ORIGINS` to a comma-separated list of allowed website origins when the frontend is not on localhost:3000.

The starter website is served by the same app at `http://localhost:8000/`. Open that URL, choose a browser-accessible camera, allow permission, and press **Start camera**. The browser sends one frame per second and displays events received over the WebSocket. Browser camera access requires localhost or HTTPS. To use a separate physical camera, connect it to the computer/browser or run an edge client that posts frames to the camera endpoint.

SAM 2.1 is optional at startup. To enable liquid-mask proposals, install a CUDA-compatible PyTorch build and SAM 2.1 for the host, then set `SAM2_CONFIG`, `SAM2_CHECKPOINT`, and `SAM2_DEVICE`. Without it, camera ingest and color inspection still work and the mask result reports that segmentation is unavailable.

## Live camera to frontend flow

1. The website opens `ws://<backend-host>:8000/ws/frontend/<camera_id>` and keeps the connection open. Each camera has its own subscription channel.
2. The camera/edge process posts a JPEG, PNG, or WebP to `POST /v1/cameras/{camera_id}/frames` as multipart form data. Required fields are `image`, `source_id`, `pts_seconds`, `width`, and `height`. Optional fields include same-frame `detections_json`, recipe IDs, vial index, and an expected clean-recipe RGB reference.
3. The backend returns an acknowledgement and publishes a `camera.frame` JSON event to connected website clients. The event includes the received frame as base64, source frame ID and PTS, inspection decision, and per-proposal masks. The frontend can render it immediately without polling.

The included dashboard is a starter interface for this flow. It can use a calibrated RGB reference and a liquid ROI to show color alerts. It does not include a vial/liquid detector or a calibrated reference database; the detection model and recipe reference values must be connected before treating alerts or masks as production inspection results.

Example form fields:

```text
source_id=cam1-000184
pts_seconds=7.44
width=640
height=480
detections_json=[{"xyxy":[230,160,410,450],"class_id":1,"label":"liquid","confidence":0.94,"source":"edge-detector"}]
current_recipe_id=7
previous_recipe_id=6
vial_index=1
is_first_vial_after_change=true
expected_color_rgb_json=[150,178,205]
color_roi_xyxy_json=[260,270,370,420]
```

The example RGB and coordinates are illustrative only. The camera/edge client should supply calibrated recipe references and a liquid-only vial ROI. If an observed RGB value is not provided but `color_roi_xyxy_json` is, the backend measures the median RGB from that region. `color_inspection` reports CIEDE2000 ΔE against the clean reference, alerts above the research-plan threshold of 2.3, and only names the previous recipe as a plausible carryover cause when the frame is explicitly marked as the first vial after a change. It reports α as an unestimated [0, 1] range because the research found carryover variability; it does not claim to estimate α from one frame.

## API endpoints

- `GET /health` reports service and connected frontend counts.
- `POST /v1/cameras/{camera_id}/frames` receives camera frames and broadcasts them to the matching WebSocket channel.
- `WS /ws/frontend/{camera_id}` streams live frame and inspection events to the website.
- `POST /v1/analyze` and `POST /v1/analyze-frame` provide synchronous analysis without broadcasting.
- `POST /v1/streams/{stream_id}/reset` clears the in-memory sequence evidence window.

## Data and decision boundaries

- Camera frame dimensions must match the submitted width and height; PTS stays in source-video seconds.
- Detection boxes use pixel `xyxy` with exclusive right and bottom edges. Class 1 is treated as a liquid prompt; class 0 vial detections are excluded from pool-mask prompts.
- Mask outputs are full-frame grayscale PNGs with binary 0/255 values and remain aligned to source coordinates.
- A missing detection is an explicit no-proposal. Pool presence and sample usability remain separate fields.
- Color comparison requires an approved clean reference under matched camera/lighting settings. The reference CSVs describe color names but do not define calibrated RGB targets, so the service does not invent them.
- Live WebSocket subscriptions and sequence state are in process memory. For multiple API workers or multiple servers, use a shared pub/sub service and shared state store before deployment.
- Video 5 and its labels are references for integration; the research plan warns that timestamps drift due to 24.7 vs 30 fps. Do not treat those timing labels as ground-truth camera timestamps.

## Source context included

The `data/` folder contains the supplied manifest, Video 5 frame labels and vial recipes, and image index CSVs for later dataset-backed screens. These data are reference context, not approved recipe color calibration values.
