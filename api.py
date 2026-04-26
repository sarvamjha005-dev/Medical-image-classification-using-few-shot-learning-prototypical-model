import os

import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


ALL_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_model_path():
    candidates = [
        "hybrid_siamese.pth",
        "model_lr0.001_bs32.pth",
        "model_lr0.001_bs16.pth",
        "model_lr0.0005_bs32.pth",
        "model_lr0.0005_bs16.pth",
        "model_lr0.0001_bs32.pth",
        "model_lr0.0001_bs16.pth",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return "hybrid_siamese.pth"


MODEL_PATH = os.environ.get("MODEL_PATH", default_model_path())
DEFAULT_THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", "0.30"))
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "5"))


class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.embedding = nn.Linear(512, 128)
        self.classifier = nn.Linear(128, len(ALL_LABELS))

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        emb = self.embedding(x)
        out = self.classifier(emb)
        return emb, out


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def load_trained_model(model_path):
    model = HybridModel().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(model, image, threshold=0.3, top_k=5):
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        _, logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    ranked = sorted(
        [{"label": ALL_LABELS[i], "score": float(score)} for i, score in enumerate(probs)],
        key=lambda item: item["score"],
        reverse=True,
    )

    filtered = [item for item in ranked if item["score"] >= threshold]
    predictions = filtered[:top_k] if filtered else ranked[:top_k]
    top_prediction = predictions[0]

    return {
        "predictions": predictions,
        "top_prediction": top_prediction,
        "all_scores": ranked,
    }


app = Flask(__name__)
model = load_trained_model(MODEL_PATH)


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Class Prediction API</title>
  <style>
    :root {
      --bg: #08111f;
      --bg-2: #143350;
      --glass: rgba(9, 20, 36, 0.74);
      --line: rgba(153, 223, 255, 0.16);
      --text: #eef7ff;
      --muted: #9eb5ca;
      --accent: #73e8ff;
      --accent-2: #ffb66d;
      --good: #84f1c0;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Georgia, serif;
      color: var(--text);
      background:
        radial-gradient(circle at 15% 15%, rgba(115, 232, 255, 0.14), transparent 24%),
        radial-gradient(circle at 85% 15%, rgba(255, 182, 109, 0.14), transparent 26%),
        linear-gradient(135deg, var(--bg), var(--bg-2));
    }

    .page {
      width: min(1100px, calc(100vw - 32px));
      margin: 24px auto;
      padding: 24px;
      border-radius: 28px;
      border: 1px solid var(--line);
      background: rgba(4, 10, 18, 0.48);
      backdrop-filter: blur(18px);
    }

    .hero, .results {
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 24px;
    }

    .results {
      margin-top: 24px;
    }

    .card, .panel {
      padding: 24px;
      border-radius: 24px;
      border: 1px solid var(--line);
      background: var(--glass);
    }

    .badge {
      display: inline-block;
      padding: 8px 14px;
      border-radius: 999px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--accent);
      background: rgba(115, 232, 255, 0.08);
      border: 1px solid rgba(115, 232, 255, 0.18);
    }

    h1 {
      margin: 16px 0 12px;
      font-size: clamp(2.5rem, 4.4vw, 4.9rem);
      line-height: 0.95;
      letter-spacing: -0.05em;
    }

    .lead, .hint, .status, .placeholder {
      color: var(--muted);
    }

    .lead {
      margin: 0;
      line-height: 1.7;
      max-width: 58ch;
    }

    .stats {
      margin-top: 22px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }

    .stat {
      padding: 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
    }

    .stat strong {
      display: block;
      margin-bottom: 6px;
      font-size: 1.08rem;
    }

    .field {
      margin-bottom: 16px;
    }

    label {
      display: block;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 0.92rem;
    }

    input[type="file"], input[type="number"], button {
      width: 100%;
      border-radius: 16px;
      font-size: 0.98rem;
    }

    input[type="file"], input[type="number"] {
      padding: 14px 16px;
      color: var(--text);
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
    }

    button {
      padding: 14px 18px;
      border: 0;
      font-weight: 700;
      cursor: pointer;
      color: #04111a;
      background: linear-gradient(135deg, var(--accent-2), var(--accent));
    }

    .status {
      margin-top: 10px;
      min-height: 22px;
    }

    .panel {
      min-height: 340px;
    }

    .placeholder {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 290px;
      text-align: center;
      line-height: 1.7;
    }

    .preview img {
      width: 100%;
      max-height: 420px;
      object-fit: contain;
      border-radius: 18px;
      background: rgba(255,255,255,0.03);
    }

    .winner {
      margin-bottom: 18px;
      padding: 18px;
      border-radius: 20px;
      background: linear-gradient(135deg, rgba(132, 241, 192, 0.14), rgba(115, 232, 255, 0.08));
      border: 1px solid rgba(132, 241, 192, 0.2);
    }

    .winner h2 {
      margin: 6px 0;
      font-size: 2rem;
    }

    .list {
      display: grid;
      gap: 12px;
    }

    .row {
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
    }

    .row-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 10px;
    }

    .bar {
      height: 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.07);
      overflow: hidden;
    }

    .bar span {
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--accent-2), var(--accent));
      border-radius: inherit;
    }

    @media (max-width: 940px) {
      .hero, .results, .stats {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <article class="card">
        <div class="badge">Simple Prediction API</div>
        <h1>Upload one image, get predicted classes.</h1>
        <p class="lead">
          This API accepts a single image and returns ranked class scores from the loaded classifier model.
        </p>
        <div class="stats">
          <div class="stat">
            <strong>/predict</strong>
            <span>Single image in, ranked classes out.</span>
          </div>
          <div class="stat">
            <strong>__MODEL_NAME__</strong>
            <span>Weight file loaded at startup.</span>
          </div>
          <div class="stat">
            <strong>14 classes</strong>
            <span>NIH chest X-ray labels returned by the API.</span>
          </div>
        </div>
      </article>

      <aside class="card">
        <form id="predictForm">
          <div class="field">
            <label for="imageInput">Upload image</label>
            <input id="imageInput" name="image" type="file" accept="image/*" required />
          </div>

          <div class="field">
            <label for="thresholdInput">Threshold</label>
            <input id="thresholdInput" name="threshold" type="number" min="0" max="1" step="0.01" value="__THRESHOLD__" />
          </div>

          <div class="field">
            <label for="topKInput">Top K</label>
            <input id="topKInput" name="top_k" type="number" min="1" max="14" step="1" value="__TOP_K__" />
          </div>

          <button type="submit">Predict</button>
          <div class="hint">JSON endpoint: <code>POST /predict</code> with form field <code>image</code>.</div>
        </form>
        <div class="status" id="status">Waiting for image upload.</div>
      </aside>
    </section>

    <section class="results">
      <div class="panel preview" id="previewPanel">
        <div class="placeholder">The uploaded image preview will appear here.</div>
      </div>
      <div class="panel" id="resultsPanel">
        <div class="placeholder">Predicted classes will appear here after inference.</div>
      </div>
    </section>
  </main>

  <script>
    const form = document.getElementById("predictForm");
    const imageInput = document.getElementById("imageInput");
    const previewPanel = document.getElementById("previewPanel");
    const resultsPanel = document.getElementById("resultsPanel");
    const statusBox = document.getElementById("status");

    imageInput.addEventListener("change", () => {
      const file = imageInput.files[0];
      if (!file) {
        previewPanel.innerHTML = '<div class="placeholder">The uploaded image preview will appear here.</div>';
        return;
      }
      previewPanel.innerHTML = `<img src="${URL.createObjectURL(file)}" alt="Preview" />`;
      statusBox.textContent = `Ready to predict: ${file.name}`;
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      const file = imageInput.files[0];
      if (!file) {
        statusBox.textContent = "Please upload an image.";
        return;
      }

      const formData = new FormData();
      formData.append("image", file);
      formData.append("threshold", document.getElementById("thresholdInput").value);
      formData.append("top_k", document.getElementById("topKInput").value);

      statusBox.textContent = "Running inference...";
      resultsPanel.innerHTML = '<div class="placeholder">Scoring classes...</div>';

      try {
        const response = await fetch("/predict", {
          method: "POST",
          body: formData
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Prediction failed.");
        }

        const rows = data.predictions.map((item) => `
          <div class="row">
            <div class="row-head">
              <strong>${item.label}</strong>
              <span>${(item.score * 100).toFixed(2)}%</span>
            </div>
            <div class="bar"><span style="width:${Math.max(item.score * 100, 3)}%"></span></div>
          </div>
        `).join("");

        resultsPanel.innerHTML = `
          <div class="winner">
            <div style="color:var(--muted)">Top prediction</div>
            <h2>${data.top_prediction.label}</h2>
            <div>${(data.top_prediction.score * 100).toFixed(2)}% confidence</div>
          </div>
          <div class="list">${rows}</div>
        `;

        statusBox.textContent = "Prediction complete.";
      } catch (error) {
        statusBox.textContent = "Prediction failed.";
        resultsPanel.innerHTML = `<div class="placeholder">${error.message}</div>`;
      }
    });
  </script>
</body>
</html>
"""


def parse_top_k(value):
    try:
        return min(max(1, int(value)), len(ALL_LABELS))
    except (TypeError, ValueError):
        return DEFAULT_TOP_K


def parse_threshold(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD
    return min(max(parsed, 0.0), 1.0)


@app.get("/")
def home():
    return (
        HTML_PAGE
        .replace("__MODEL_NAME__", os.path.basename(MODEL_PATH))
        .replace("__THRESHOLD__", str(DEFAULT_THRESHOLD))
        .replace("__TOP_K__", str(DEFAULT_TOP_K))
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": os.path.basename(MODEL_PATH),
            "class_count": len(ALL_LABELS),
            "mode": "single_image_classifier",
        }
    )


@app.post("/predict")
def predict():
    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"error": "Missing image upload."}), 400

    threshold = parse_threshold(request.form.get("threshold"))
    top_k = parse_top_k(request.form.get("top_k"))

    try:
        image = Image.open(image_file.stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Uploaded file is not a valid image."}), 400

    try:
        result = predict_image(model, image, threshold=threshold, top_k=top_k)
    except Exception as exc:
        return jsonify({"error": "Prediction failed.", "details": str(exc)}), 500

    return jsonify(
        {
            "success": True,
            "model": os.path.basename(MODEL_PATH),
            "threshold": threshold,
            "top_k": top_k,
            **result,
        }
    )


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5080"))
    print(f"Serving Class Prediction API on http://{host}:{port}")
    print(f"Loaded model weights from: {MODEL_PATH}")
    app.run(host=host, port=port, debug=False)
