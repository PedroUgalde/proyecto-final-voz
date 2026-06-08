"""
Interfaz web local para probar el clasificador de voz.

Uso:
    conda activate K-means-clustering
    python app/web_app.py
"""

import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CTC_MODEL_PATH, TARGET_WORDS
from src.dataset import extract_mfcc_file
from src.classic.kmeans_classifier import KMeansWordClassifier
from src.classic.gmm_classifier import GMMWordClassifier
from src.classic.dtw import dtw_predict
from src.config import KMEANS_MODEL_PATH, GMM_MODEL_PATH, DTW_TEMPLATES_PATH
from src.predict_ctc import predict_greedy, match_to_vocabulary
from src.train_ctc import load_ctc_model
from src.predict_beam_search import predict_beam_search

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Reconocimiento de Voz - Speech Commands</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; background: #0f172a; color: #e2e8f0; }
    h1 { color: #38bdf8; }
    .card { background: #1e293b; border-radius: 12px; padding: 24px; margin: 16px 0; box-shadow: 0 4px 20px rgba(0,0,0,.3); }
    button { background: #38bdf8; color: #0f172a; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 16px; margin: 8px 4px; }
    button:hover { background: #7dd3fc; }
    button.recording { background: #ef4444; color: white; animation: pulse 1s infinite; }
    @keyframes pulse { 50% { opacity: 0.7; } }
    .result { font-size: 2em; text-align: center; padding: 20px; color: #4ade80; }
    select { padding: 8px; border-radius: 6px; background: #334155; color: #e2e8f0; border: 1px solid #475569; }
    .words { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .word { background: #334155; padding: 6px 12px; border-radius: 20px; font-size: 14px; }
  </style>
</head>
<body>
  <h1>🎤 Reconocimiento de Voz</h1>
  <p>Sistema Bi-LSTM + CTC | Speech Commands Dataset</p>

  <div class="card">
    <label>Modelo: </label>
    <select id="model">
      <option value="ctc_greedy">CTC + Greedy Search</option>
      <option value="ctc_beam">CTC + Beam Search</option>
      <option value="kmeans">K-Means</option>
      <option value="gmm">GMM</option>
      <option value="dtw">DTW</option>
    </select>
    <br><br>
    <input type="file" id="file" accept="audio/*">
    <button onclick="upload()">Subir y clasificar</button>
    <button id="recBtn" onclick="toggleRecord()">🎙 Grabar</button>
    <div class="result" id="result">—</div>
  </div>

  <div class="card">
    <strong>Palabras reconocibles:</strong>
    <div class="words">
      {% for w in words %}<span class="word">{{ w }}</span>{% endfor %}
    </div>
  </div>

  <script>
    let mediaRecorder, chunks = [], recording = false;

    async function toggleRecord() {
      const btn = document.getElementById('recBtn');
      if (!recording) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        chunks = [];
        mediaRecorder.ondataavailable = e => chunks.push(e.data);
        mediaRecorder.onstop = async () => {
          const blob = new Blob(chunks, { type: 'audio/wav' });
          await classify(blob);
        };
        mediaRecorder.start();
        recording = true;
        btn.textContent = '⏹ Detener';
        btn.classList.add('recording');
      } else {
        mediaRecorder.stop();
        recording = false;
        btn.textContent = '🎙 Grabar';
        btn.classList.remove('recording');
      }
    }

    async function upload() {
      const f = document.getElementById('file').files[0];
      if (f) await classify(f);
    }

    async function classify(blob) {
      document.getElementById('result').textContent = 'Procesando...';
      const fd = new FormData();
      fd.append('audio', blob, 'audio.wav');
      fd.append('model', document.getElementById('model').value);
      const r = await fetch('/predict', { method: 'POST', body: fd });
      const data = await r.json();
      document.getElementById('result').textContent = data.prediction || data.error;
    }
  </script>
</body>
</html>
"""


def _load_models():
    models = {}
    if CTC_MODEL_PATH.exists():
        models["ctc"], _ = load_ctc_model(CTC_MODEL_PATH)
    if KMEANS_MODEL_PATH.exists():
        models["kmeans"] = KMeansWordClassifier.load(KMEANS_MODEL_PATH)
    if GMM_MODEL_PATH.exists():
        models["gmm"] = GMMWordClassifier.load(GMM_MODEL_PATH)
    if DTW_TEMPLATES_PATH.exists():
        import pickle
        with open(DTW_TEMPLATES_PATH, "rb") as f:
            models["dtw"] = pickle.load(f)
    return models


MODELS = {}


@app.route("/")
def index():
    return render_template_string(HTML, words=TARGET_WORDS)


@app.route("/predict", methods=["POST"])
def predict():
    global MODELS
    if not MODELS:
        MODELS = _load_models()

    model_type = request.form.get("model", "ctc_greedy")
    if "audio" not in request.files:
        return jsonify({"error": "No se recibió audio"}), 400

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        request.files["audio"].save(tmp.name)
        path = tmp.name

    try:
        if model_type == "ctc_greedy":
            if "ctc" not in MODELS:
                return jsonify({"error": "Modelo CTC no entrenado. Ejecuta el notebook primero."}), 400
            raw = predict_greedy(MODELS["ctc"], path)
            pred = match_to_vocabulary(raw)
        elif model_type == "ctc_beam":
            if "ctc" not in MODELS:
                return jsonify({"error": "Modelo CTC no entrenado."}), 400
            pred = predict_beam_search(MODELS["ctc"], path)
        elif model_type == "kmeans":
            if "kmeans" not in MODELS:
                return jsonify({"error": "Modelo K-Means no entrenado."}), 400
            mfcc = extract_mfcc_file(path)
            pred = MODELS["kmeans"].predict(mfcc)
        elif model_type == "gmm":
            if "gmm" not in MODELS:
                return jsonify({"error": "Modelo GMM no entrenado."}), 400
            mfcc = extract_mfcc_file(path)
            pred = MODELS["gmm"].predict(mfcc)
        elif model_type == "dtw":
            if "dtw" not in MODELS:
                return jsonify({"error": "Plantillas DTW no entrenadas."}), 400
            mfcc = extract_mfcc_file(path)
            pred = dtw_predict(mfcc, MODELS["dtw"])
        else:
            return jsonify({"error": "Modelo desconocido"}), 400

        return jsonify({"prediction": pred, "model": model_type})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Cargando modelos...")
    MODELS = _load_models()
    print(f"Modelos disponibles: {list(MODELS.keys())}")
    print("Abre http://127.0.0.1:5000 en tu navegador")
    app.run(debug=False, port=5000)
