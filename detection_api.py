from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from detection_processing import (
    read_and_compile_model,
    infer_model,
    draw_detections,
    stream_video_frames,
    load_labels,
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
        <title>Ore Detection</title>
        <style>
          body { font-family: system-ui, Arial, sans-serif; max-width: 1900px;
                 margin: 2rem auto; padding: 0 1rem; color: #222; background-color:#292929}
          h1 { margin-bottom: .25rem; color:white; }
          .card { border: 1px solid #ddd; border-radius: 10px; padding: 1.25rem;
                  box-shadow: 0 1px 4px rgba(0,0,0,.05); }
          .row { display: flex; gap: .75rem; align-items: center; flex-wrap: wrap; }
          input[type=file] { flex: 1; min-width: 220px; color:white }
          button { background: #166534; color: #fff; border: 0; border-radius: 6px;
                   padding: .55rem 1.1rem; font-size: 1rem; cursor: pointer; }
          button:disabled { opacity: .5; cursor: default; }
          button.secondary { background: #555; }
          .progress-wrap { display: none; margin-top: 1rem; }
          .progress { height: 10px; background: #eee; border-radius: 6px; overflow: hidden; }
          .progress-bar { height: 100%; width: 0%; background: #16a34a; transition: width .15s; }
          .progress-text { font-size: .8rem; color: #555; margin-top: .25rem; }
          .spinner { display: none; margin-top: 1rem; }
          .spinner span { display: inline-block; width: 22px; height: 22px;
            border: 3px solid #ccc; border-top-color: #16a34a; border-radius: 50%;
            animation: spin .8s linear infinite; vertical-align: middle; }
          @keyframes spin { to { transform: rotate(360deg); } }
          #result { margin-top: 0; }
          #result img { border: 1px solid #ddd; border-radius: 8px;
                        max-width: none; width: max-content; height: auto; display: block; }
          .layout { display: flex; gap: 1.5rem; align-items: flex-start; margin-top: 1.25rem; }
          .sidebar { flex: 0 0 500px; position: sticky; top: 1rem; }
          .preview { flex: 1; min-width: 0; margin:0 }
          table { border-collapse: collapse; margin-top: .75rem; width:500px; }
          th, td { border: 1px solid #ddd; padding: .4rem .7rem; }
          #error { display: none; color: #b91c1c; margin-top: 1rem; font-weight: 600; }
          .muted { color: #bab8b8; font-size: .85rem; }
          #another { display: none; margin-top: 1rem; }
        </style>
      </head>
      <body>
        <h1>Ore Detection</h1>
        <p class="muted">Upload an image or video to detect ores in your browser.</p>
        <div class="layout">
          <div class="sidebar">
            <div class="card">
              <div class="row">
                <input type="file" id="file" accept="image/*,video/*" required>
                <button id="upload">Detect</button>
              </div>

              <div class="progress-wrap" id="progressWrap">
                <div class="progress"><div class="progress-bar" id="progressBar"></div></div>
                <div class="progress-text" id="progressText" style="color:white;">0%</div>
              </div>

              <div class="spinner" id="spinner"><span></span> Processing&hellip;</div>
              <div id="error"></div>
              <button id="another" class="secondary" onclick="resetUI()">Upload another</button>
              <table><thead><tr><th style="font-size:14px;color:white;">Label</th><th style="font-size:14px;color:white;">Confidence</th>
                <th style="font-size:14px;color:white;">BBox (x1, y1, x2, y2)</th></tr></thead>
                <tbody id="detBody"></tbody></table>
            </div>
          </div>
          <div class="preview">
            <div id="result"></div>
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
          let stopVideoPoll = null;

          function esc(s) {
            return String(s).replace(/[&<>"']/g, c => (
              {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
          }
          function showError(msg) {
            spinner.style.display = "none";
            errorBox.textContent = msg;
            errorBox.style.display = "block";
          }
          function buildRows(detections) {
            return (detections || []).map(d => (
              "<tr><td style='color:white;'>" + esc(d.label) + "</td><td style='color:white;'>" +
              Number(d.confidence).toFixed(3) + "</td><td style='color:white;'>[" +
              d.bbox.map(v => Number(v).toFixed(1)).join(", ") + "]</td></tr>"
            )).join("");
          }
          function pollVideo(streamId) {
            const tick = () => {
              fetch("/video_detections/" + streamId)
                .then(r => {
                  if (!r.ok) throw new Error("gone");
                  return r.json();
                })
                .then(j => {
                  document.getElementById("detBody").innerHTML = buildRows(j.detections);
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
            result.innerHTML = "";
            document.getElementById("detBody").innerHTML = "";
            errorBox.style.display = "none";
            progressWrap.style.display = "none";
            spinner.style.display = "none";
            anotherBtn.style.display = "none";
            fileInput.value = "";
            uploadBtn.disabled = false;
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
            document.getElementById("detBody").innerHTML = "";
            anotherBtn.style.display = "none";
            uploadBtn.disabled = true;
            xhr.send(fd);
          };

          function renderResult(data) {
            anotherBtn.style.display = "inline-block";
            uploadBtn.disabled = false;
            if (data.type === "image") {
              if (stopVideoPoll) stopVideoPoll();
              document.getElementById("detBody").innerHTML = buildRows(data.detections);
              result.innerHTML =
                '<img src="data:image/jpeg;base64,' + data.media_base64 + '">';
            } else if (data.type === "video") {
              document.getElementById("detBody").innerHTML = "";
              result.innerHTML =
                '<img id="stream" src="/video_stream/' + data.stream_id + '">' +
                '<p class="muted">This is a live MJPEG feed &mdash; no seek or ' +
                "play/pause controls.</p>";
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
