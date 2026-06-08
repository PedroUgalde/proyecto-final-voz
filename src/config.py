"""Configuración global del proyecto de reconocimiento de voz."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"

# Palabras del desafío clásico de Speech Commands (10 comandos)
TARGET_WORDS = [
    "yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go",
]

# Parámetros de audio
SAMPLE_RATE = 16000
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10
N_FFT = 512
N_MELS = 40
N_MFCC = 13
PREEMPHASIS = 0.97

# CTC: caracteres + blank (índice 0)
CHARS = ["<blank>"] + sorted(set("".join(TARGET_WORDS)))
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}

# Entrenamiento CTC
HIDDEN_SIZE = 128
NUM_LSTM_LAYERS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 15
SAMPLES_PER_CLASS = 300  # submuestreo para proyecto escolar

# Rutas de modelos
CTC_MODEL_PATH = MODELS_DIR / "ctc_phrase_model.pth"
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_model.pkl"
GMM_MODEL_PATH = MODELS_DIR / "gmm_model.pkl"
DTW_TEMPLATES_PATH = MODELS_DIR / "dtw_templates.pkl"
