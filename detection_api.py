from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from detection_processing import (
    read_and_compile_model,
    infer_model,
    draw_detections,
    stream_video_frames,
    load_labels,
    get_rock_info,
)
import os
import asyncio
import tempfile
import urllib.request
import base64
import uuid
import cv2
import uvicorn


MODEL_XML = os.path.abspath('./ore-detection-model/deployment/Detection/model/model.xml')
MODEL_BIN = os.path.abspath('./ore-detection-model/deployment/Detection/model/model.bin')
CONFIG = os.path.abspath('./ore-detection-model/deployment/Detection/model/config.json')

dt_model = read_and_compile_model(MODEL_XML,MODEL_BIN)
labels = load_labels(CONFIG)

app = FastAPI()

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv")
STREAMS = {}


def is_video(filename: str, content_type: str = "") -> bool:
    name = (filename or "").lower()
    if name.endswith(VIDEO_EXTENSIONS):
        return True
    return content_type.startswith("video/")


@app.get("/", response_class=HTMLResponse)
def home():
    html = """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>MINE ORE DETECTION SYSTEM</title>
        <style>
          * { margin: 0; padding: 0; box-sizing: border-box; }
          body {
            font-family: 'Segoe UI', system-ui, Arial, sans-serif;
            font-size: 19px;
            background: #0a0a14;
            color: #c8c8d0;
            min-height: 100vh;
          }
          .header {
            background: linear-gradient(180deg, #141428 0%, #0f0f20 100%);
            border-bottom: 3px solid #f59e0b;
            padding: 0.75rem 2rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            box-shadow: 0 2px 20px rgba(245, 158, 11, 0.08);
          }
          .header-icon {
            font-size: 1.6rem;
            line-height: 1;
          }
          .header h1 {
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 1.15rem;
            font-weight: 700;
            color: #f59e0b;
            letter-spacing: 2px;
            text-transform: uppercase;
          }
          .header .sub {
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 0.7rem;
            color: #6b7280;
            letter-spacing: 1px;
            margin-left: auto;
          }
          .container {
            max-width: 1900px;
            margin: 1.25rem auto;
            padding: 0 1.25rem;
          }
          .layout {
            display: flex;
            gap: 1.5rem;
            align-items: flex-start;
          }
          .sidebar {
            flex: 0 0 520px;
            position: sticky;
            top: 1rem;
          }
          .panel {
            background: #12121e;
            border: 1px solid #2a2a44;
            border-radius: 4px;
            padding: 1.25rem;
            box-shadow: 0 0 0 1px rgba(245, 158, 11, 0.05), 0 4px 24px rgba(0,0,0,0.4);
          }
          .panel-title {
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 0.7rem;
            color: #f59e0b;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #2a2a44;
          }
          .row { display: flex; gap: .75rem; align-items: center; flex-wrap: wrap; }
          input[type=file] {
            flex: 1; min-width: 200px;
            background: #1a1a2e;
            border: 1px solid #333355;
            border-radius: 3px;
            padding: 0.5rem 0.75rem;
            color: #c8c8d0;
            font-size: 0.85rem;
            font-family: 'Courier New', 'Consolas', monospace;
          }
          input[type=file]::file-selector-button {
            background: #2a2a44;
            border: 1px solid #444466;
            border-radius: 2px;
            padding: 0.25rem 0.6rem;
            color: #c8c8d0;
            font-family: inherit;
            margin-right: 0.5rem;
            cursor: pointer;
          }
          input[type=file]::file-selector-button:hover {
            background: #333355;
          }
          button {
            background: #2a2a44;
            color: #f59e0b;
            border: 1px solid #444466;
            border-radius: 3px;
            padding: 0.55rem 1.25rem;
            font-size: 0.85rem;
            font-family: 'Courier New', 'Consolas', monospace;
            font-weight: 700;
            letter-spacing: 1px;
            cursor: pointer;
            text-transform: uppercase;
            transition: all 0.15s;
          }
          button:hover:not(:disabled) {
            background: #f59e0b;
            color: #0a0a14;
            border-color: #f59e0b;
          }
          button:disabled { opacity: .4; cursor: default; }
          button.secondary {
            background: transparent;
            border-color: #333355;
            color: #6b7280;
          }
          button.secondary:hover {
            border-color: #f59e0b;
            color: #f59e0b;
            background: transparent;
          }
          .progress-wrap { display: none; margin-top: 1rem; }
          .progress {
            height: 8px;
            background: #1a1a2e;
            border: 1px solid #333355;
            border-radius: 2px;
            overflow: hidden;
          }
          .progress-bar {
            height: 100%; width: 0%;
            background: linear-gradient(90deg, #d97706, #f59e0b);
            transition: width .15s;
          }
          .progress-text {
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: .75rem;
            color: #6b7280;
            margin-top: .3rem;
          }
          .spinner { display: none; margin-top: 1rem; }
          .spinner span {
            display: inline-block; width: 20px; height: 20px;
            border: 2px solid #2a2a44;
            border-top-color: #f59e0b;
            border-radius: 50%;
            animation: spin .8s linear infinite;
            vertical-align: middle;
            margin-right: 0.5rem;
          }
          @keyframes spin { to { transform: rotate(360deg); } }
          .spinner-text {
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: .8rem;
            color: #6b7280;
          }
          #result { margin-top: 0; display:flex; justify-content:center; flex-direction: column }
          #result img {
            border: 1px solid #2a2a44;
            border-radius: 3px;
            max-width: 100%;
            height: auto;
            display: block;
            background: #0a0a14;
          }
          .preview { flex: 1; min-width: 0; margin: 0; }
          table {
            border-collapse: collapse;
            margin-top: .75rem;
            width: 100%;
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 0.8rem;
          }
          th {
            border: 1px solid #2a2a44;
            padding: .45rem .6rem;
            color: #f59e0b;
            background: #0f0f20;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            font-size: 0.7rem;
          }
          td {
            border: 1px solid #2a2a44;
            padding: .4rem .6rem;
            color: #c8c8d0;
          }
          .label-cell {
            color: #f59e0b;
            font-weight: 700;
          }
          .conf-cell {
            color: #6ee7b7;
            text-align: right;
          }
          #error {
            display: none;
            color: #ef4444;
            margin-top: 0.75rem;
            font-weight: 600;
            font-size: 0.85rem;
            font-family: 'Courier New', 'Consolas', monospace;
            padding: 0.5rem;
            border: 1px solid #ef4444;
            border-radius: 2px;
            background: rgba(239, 68, 68, 0.08);
          }
          .muted { color: #6b7280; font-size: .8rem; }
          #another { display: none; margin-top: 0.75rem; }

          .detail-panel {
            margin-top: 1rem;
            border-top: 1px solid #2a2a44;
            padding-top: 0.75rem;
          }
          .detail-card {
            background: #0f0f20;
            border: 1px solid #2a2a44;
            border-left: 3px solid #f59e0b;
            border-radius: 2px;
            padding: 0.75rem;
            margin-top: 0.5rem;
          }
          .detail-card:first-child { margin-top: 0; }
          .detail-card .dc-header {
            display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;
            margin-bottom: 0.35rem;
          }
          .detail-card .dc-name {
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 0.85rem; color: #f59e0b; font-weight: 700;
            text-transform: uppercase; letter-spacing: 1px;
          }
          .detail-card .dc-tier {
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 0.6rem; font-weight: 700; letter-spacing: 1px;
            padding: 0.15rem 0.45rem; border-radius: 2px;
            text-transform: uppercase; border: 1px solid;
          }
          .tier-high { color: #f59e0b; border-color: #f59e0b; background: rgba(245,158,11,0.12); }
          .tier-medium { color: #3b82f6; border-color: #3b82f6; background: rgba(59,130,246,0.12); }
          .tier-low_medium { color: #14b8a6; border-color: #14b8a6; background: rgba(20,184,166,0.12); }
          .tier-low { color: #6b7280; border-color: #6b7280; background: rgba(107,114,128,0.12); }
          .tier-lowest { color: #4b5563; border-color: #4b5563; background: rgba(75,85,99,0.12); }
          .tier-unknown { color: #6b7280; border-color: #333355; background: rgba(107,114,128,0.06); }
          .detail-card .dc-rank {
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 0.65rem; color: #6b7280; margin-left: auto;
          }
          .detail-card .dc-visual {
            font-size: 0.75rem; color: #6b7280; font-style: italic;
            margin-bottom: 0.5rem; padding: 0.2rem 0.4rem;
            border-left: 2px solid #2a2a44; background: rgba(255,255,255,0.02);
          }
          .detail-card .dc-desc {
            font-size: 0.78rem; color: #9ca3af; line-height: 1.45;
            margin-bottom: 0.5rem;
          }
          .detail-card .dc-section {
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 0.65rem; color: #f59e0b; letter-spacing: 1px;
            text-transform: uppercase; margin: 0.5rem 0 0.3rem 0;
            border-bottom: 1px solid #1e1e38; padding-bottom: 0.15rem;
          }
          .detail-card .dc-char {
            font-size: 0.75rem; color: #9ca3af; line-height: 1.55;
            padding-left: 0.5rem;
          }
          .detail-card .dc-char strong {
            color: #c8c8d0; font-weight: 600;
          }
          .detail-card .dc-use {
            font-size: 0.75rem; color: #9ca3af; line-height: 1.6;
            padding-left: 0.5rem; list-style: none;
          }
          .detail-card .dc-use li::before { content: " -> "; color: #f59e0b; }
          .detail-card .dc-price {
            font-size: 0.75rem; color: #6ee7b7; line-height: 1.6;
            padding-left: 0.5rem; font-family: 'Courier New', 'Consolas', monospace;
          }

          .status-bar {
            display: flex;
            gap: 1.5rem;
            margin-top: 0.75rem;
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 0.7rem;
            color: #6b7280;
          }
          .status-bar .val {
            color: #f59e0b;
          }
          .divider {
            height: 1px;
            background: #2a2a44;
            margin: 1rem 0;
          }
        </style>
      </head>
      <body>
        <div class="header">
          <span class="header-icon">&#9881;</span>
          <h1>Mine Ore Detection System</h1>
          <span class="sub">v2.1 &nbsp;|&nbsp; INDUSTRIAL VISION MODULE</span>
        </div>

        <div class="container">
          <div class="layout">
            <div class="sidebar">
              <div class="panel">
                <div class="panel-title">&#9654; Control Panel</div>
                <div class="row">
                  <input type="file" id="file" accept="image/*,video/*" required>
                  <button id="upload">Detect</button>
                </div>

                <div class="progress-wrap" id="progressWrap">
                  <div class="progress"><div class="progress-bar" id="progressBar"></div></div>
                  <div class="progress-text" id="progressText">0%</div>
                </div>

                <div class="spinner" id="spinner">
                </div>
                <div id="error"></div>
                <button id="another" class="secondary" onclick="resetUI()">Upload Another</button>

                <div class="divider"></div>

                <div class="panel-title">&#9654; Detection Results</div>
                <table>
                  <thead>
                    <tr>
                      <th>Label</th>
                      <th>Conf.</th>
                      <th>BBox (x1, y1, x2, y2)</th>
                    </tr>
                  </thead>
                  <tbody id="detBody"></tbody>
                </table>

                <div class="detail-panel" id="detailPanel"></div>

                <div class="status-bar" id="statusBar">
                  <span>STATUS: <span class="val" id="statusText">STANDBY</span></span>
                  <span>DETECTIONS: <span class="val" id="countText">0</span></span>
                </div>
              </div>
            </div>
            <div class="preview">
              <div class="panel">
                <div class="panel-title">&#9654; Inspection View</div>
                <div id="result">
                  <p class="muted" style="padding: 2rem 0; text-align: center;">
                    Upload an image or video to begin ore analysis.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <script>
          const fileInput = document.getElementById("file");
          const uploadBtn = document.getElementById("upload");
          const progressWrap = document.getElementById("progressWrap");
          const progressBar = document.getElementById("progressBar");
          const progressText = document.getElementById("progressText");
          const spinner = document.getElementById("spinner");
          const result = document.getElementById("result");
          const errorBox = document.getElementById("error");
          const anotherBtn = document.getElementById("another");
          const detBody = document.getElementById("detBody");
          const detailPanel = document.getElementById("detailPanel");
          const statusText = document.getElementById("statusText");
          const countText = document.getElementById("countText");
          let stopVideoPoll = null;

          function esc(s) {
            return String(s).replace(/[&<>"']/g, c => (
              {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
          }
          function showError(msg) {
            spinner.style.display = "none";
            errorBox.textContent = msg;
            errorBox.style.display = "block";
            statusText.textContent = "ERROR";
          }
          function buildRows(detections) {
            return (detections || []).map((d, i) => (
              "<tr>" +
                "<td class='label-cell'>" + esc(d.label) + "</td>" +
                "<td class='conf-cell'>" + Number(d.confidence).toFixed(3) + "</td>" +
                "<td>[" + d.bbox.map(v => Number(v).toFixed(1)).join(", ") + "]</td>" +
              "</tr>"
            )).join("");
          }
          function buildDetailPanel(detections) {
            if (!detections || detections.length === 0) return "";
            let html = '<div class="panel-title">\\u25b4 Rock Details</div>';
            detections.forEach(d => {
              const r = d.rock_info || {};
              const tier = r.value_tier || "unknown";
              const tierClass = "tier-" + tier;
              html += '<div class="detail-card">';

              html += '<div class="dc-header">';
              html += '<span class="dc-name">' + esc(r.name || d.label) + '</span>';
              html += '<span class="dc-tier ' + tierClass + '">' + esc(tier.replace("_", " ")) + '</span>';
              if (r.value_rank !== undefined && r.value_rank !== "-") {
                html += '<span class="dc-rank">RANK #' + r.value_rank + '</span>';
              }
              html += '</div>';

              if (r.visual_cue) {
                html += '<div class="dc-visual">' + esc(r.visual_cue) + '</div>';
              }

              if (r.description) {
                html += '<div class="dc-desc">' + esc(r.description) + '</div>';
              }

              const chars = r.key_characteristics || {};
              if (chars.appearance || chars.texture || chars.origin) {
                html += '<div class="dc-section">Key Characteristics</div>';
                if (chars.appearance) html += '<div class="dc-char"><strong>Appearance:</strong> ' + esc(chars.appearance) + '</div>';
                if (chars.texture) html += '<div class="dc-char"><strong>Texture:</strong> ' + esc(chars.texture) + '</div>';
                if (chars.origin) html += '<div class="dc-char"><strong>Origin:</strong> ' + esc(chars.origin) + '</div>';
              }

              const uses = r.common_uses || [];
              if (uses.length > 0) {
                html += '<div class="dc-section">Common Uses</div>';
                html += '<ul class="dc-use">';
                uses.forEach(u => { html += '<li>' + esc(u) + '</li>'; });
                html += '</ul>';
              }

              const prices = r.price_usd || {};
              const priceKeys = Object.keys(prices).filter(k => Array.isArray(prices[k]));
              if (priceKeys.length > 0) {
                html += '<div class="dc-section">Estimated Price (USD)</div>';
                priceKeys.forEach(k => {
                  const p = prices[k];
                  const label = k.split("_").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
                  if (p[0] === 0 && p[1] === 0) {
                    html += '<div class="dc-price">' + esc(label) + ': negligible</div>';
                  } else {
                    html += '<div class="dc-price">' + esc(label) + ': $' + p[0] + ' &ndash; $' + p[1] + '</div>';
                  }
                });
              }

              html += '</div>';
            });
            return html;
          }
          function pollVideo(streamId) {
            const tick = () => {
              fetch("/video_detections/" + streamId)
                .then(r => {
                  if (!r.ok) throw new Error("gone");
                  return r.json();
                })
                .then(j => {
                  detBody.innerHTML = buildRows(j.detections);
                  detailPanel.innerHTML = buildDetailPanel(j.detections);
                  countText.textContent = (j.detections || []).length;
                })
                .catch(() => { STREAMS_ACTIVE = false; })
                .finally(() => {
                  if (STREAMS_ACTIVE) pollVideoTimer = setTimeout(tick, 500);
                });
            };
            let STREAMS_ACTIVE = true;
            let pollVideoTimer = null;
            stopVideoPoll = () => { STREAMS_ACTIVE = false; if (pollVideoTimer) clearTimeout(pollVideoTimer); stopVideoPoll = null; };
            tick();
          }
          function resetUI() {
            if (stopVideoPoll) stopVideoPoll();
            result.innerHTML =
              '<p class="muted" style="padding: 2rem 0; text-align: center;">' +
              'Upload an image or video to begin ore analysis.</p>';
            detBody.innerHTML = "";
            detailPanel.innerHTML = "";
            errorBox.style.display = "none";
            progressWrap.style.display = "none";
            spinner.style.display = "none";
            anotherBtn.style.display = "none";
            fileInput.value = "";
            uploadBtn.disabled = false;
            statusText.textContent = "STANDBY";
            countText.textContent = "0";
          }

          uploadBtn.onclick = () => {
            const file = fileInput.files[0];
            if (!file) { showError("Please choose an image or video file."); return; }

            const fd = new FormData();
            fd.append("file", file);

            const xhr = new XMLHttpRequest();
            xhr.open("POST", "/detect");

            xhr.upload.onprogress = (e) => {
              if (e.lengthComputable) {
                const pct = Math.round(e.loaded / e.total * 100);
                progressBar.style.width = pct + "%";
                progressText.textContent = pct + "%";
              }
            };
            xhr.onload = () => {
              spinner.style.display = "none";
              if (xhr.status !== 200) {
                showError("Upload failed (HTTP " + xhr.status + ").");
                return;
              }
              let data;
              try { data = JSON.parse(xhr.responseText); }
              catch (e) { showError("Invalid server response."); return; }
              renderResult(data);
            };
            xhr.onerror = () => showError("Network error during upload.");

            progressWrap.style.display = "block";
            progressBar.style.width = "0%";
            progressText.textContent = "0%";
            spinner.style.display = "inline-block";
            errorBox.style.display = "none";
            if (stopVideoPoll) stopVideoPoll();
            result.innerHTML = "";
            detBody.innerHTML = "";
            detailPanel.innerHTML = "";
            anotherBtn.style.display = "none";
            uploadBtn.disabled = true;
            statusText.textContent = "UPLOADING";
            xhr.send(fd);
          };

          function renderResult(data) {
            anotherBtn.style.display = "inline-block";
            uploadBtn.disabled = false;
            if (data.type === "image") {
              if (stopVideoPoll) stopVideoPoll();
              detBody.innerHTML = buildRows(data.detections);
              detailPanel.innerHTML = buildDetailPanel(data.detections);
              countText.textContent = (data.detections || []).length;
              statusText.textContent = "COMPLETE";
              result.innerHTML =
                '<img src="data:image/jpeg;base64,' + data.media_base64 + '">';
            } else if (data.type === "video") {
              detBody.innerHTML = "";
              detailPanel.innerHTML = "";
              statusText.textContent = "STREAMING";
              countText.textContent = "0";
              result.innerHTML =
                '<img id="stream" src="/video_stream/' + data.stream_id + '">' +
                '<p class="muted" style="margin-top:0.75rem;">Live MJPEG feed &mdash; ' +
                "detections update in real-time.</p>";
              pollVideo(data.stream_id);
            } else {
              showError("Unexpected response from server.");
            }
          }
        </script>
      </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/detect-ore")
async def detect_ore(image: str):
    is_url = image.startswith(("http://", "https://"))
    tmp_path = None
    try:
        if is_url:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp_path = tmp.name
            urllib.request.urlretrieve(image, tmp_path)
            img_path = tmp_path
        else:
            img_path = image

        detections = await asyncio.to_thread(
            infer_model, img_path, dt_model, labels, 0.5
        )
        for d in detections:
            d["rock_info"] = get_rock_info(d["label"])
        return {"detections": detections}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    if is_video(file.filename, file.content_type):
        stream_id = uuid.uuid4().hex
        STREAMS[stream_id] = {"path": tmp_path, "latest": []}
        return {"type": "video", "stream_id": stream_id}

    try:
        detections = await asyncio.to_thread(
            infer_model, tmp_path, dt_model, labels, 0.5
        )
        for d in detections:
            d["rock_info"] = get_rock_info(d["label"])
        img = cv2.imread(tmp_path)
        annotated = draw_detections(img, detections)
        _, buf = cv2.imencode(".jpg", annotated)
        b64 = base64.b64encode(buf).decode("utf-8")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return {
        "type": "image",
        "detections": detections,
        "media_base64": b64,
    }


@app.get("/video_stream/{stream_id}")
def video_stream(stream_id: str):
    entry = STREAMS.get(stream_id)
    if not entry or not os.path.exists(entry["path"]):
        return HTMLResponse("<p>Stream not found or expired.</p>", status_code=404)

    def mjpeg():
        try:
            for jpg, dets in stream_video_frames(entry["path"], dt_model, labels, 0.5):
                for d in dets:
                    d["rock_info"] = get_rock_info(d["label"])
                entry["latest"] = dets
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
        finally:
            if os.path.exists(entry["path"]):
                os.remove(entry["path"])
            STREAMS.pop(stream_id, None)

    return StreamingResponse(
        mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video_detections/{stream_id}")
def video_detections(stream_id: str):
    entry = STREAMS.get(stream_id)
    if not entry:
        return {"detections": []}, 404
    return {"detections": entry["latest"]}

if __name__ == "__main__":
    uvicorn.run(app="detection_api:app",port=8001,reload=True)
