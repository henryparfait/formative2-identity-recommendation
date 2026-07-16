"""
pipeline_utils.py
------------------
Shared helpers for the integrated authentication + recommendation demo.

The face and audio feature extractors here are copied FAITHFULLY from the
teammates' notebooks (P2 image, P3 audio). This matters: the demo must turn a
*new* photo / audio clip into features exactly the same way the training data
was built, or the saved models receive misaligned inputs and predict garbage.

Heavy ML libraries (cv2, tensorflow, librosa) are imported lazily inside the
functions so that --stub mode runs with no ML dependencies at all.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# FACE  (mirrors IMAGE_COLLECTION_AND_PROCESSING.ipynb)
# ----------------------------------------------------------------------------
def extract_face(image_path):
    """Detect the first face with Haar cascade, crop, resize to 224x224 (BGR)."""
    import cv2
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = cv2.resize(img[y:y + h, x:x + w], (224, 224))
    return face


_CNN = None  # cache the MobileNetV2 backbone so we build it only once


def _get_cnn():
    global _CNN
    if _CNN is None:
        from tensorflow.keras.applications import MobileNetV2
        _CNN = MobileNetV2(
            weights="imagenet", include_top=False,
            pooling="avg", input_shape=(224, 224, 3),
        )
    return _CNN


def create_embedding(face_img):
    """1280-dim MobileNetV2 average-pool embedding for a 224x224 face crop."""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    x = img_to_array(face_img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    emb = _get_cnn().predict(x, verbose=0)
    return emb.flatten()


def face_embedding_dataframe(embedding):
    """Wrap the embedding in a DataFrame with feature_0..feature_N-1 column
    names, matching how P2 trained the model (avoids sklearn feature-name
    warnings and guarantees column alignment)."""
    cols = [f"feature_{i}" for i in range(len(embedding))]
    return pd.DataFrame([embedding], columns=cols)


# ----------------------------------------------------------------------------
# AUDIO  (mirrors Audio_Proccessing_Pipeline.ipynb, Cell 8 version)
# ----------------------------------------------------------------------------
TARGET_SR = 16000


def load_audio(path):
    import librosa
    y, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return y.astype(np.float32)


def extract_audio_features(y):
    """4 spectral summaries + 13 MFCC mean/std + duration.
    Order/return is reindexed to the model bundle's feature_columns downstream."""
    import librosa
    f = {}
    f["zero_crossing_rate_mean"] = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
    f["rms_energy_mean"] = float(np.mean(librosa.feature.rms(y=y)[0]))
    f["spectral_centroid_mean"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=TARGET_SR)[0]))
    f["spectral_rolloff_mean"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=TARGET_SR)[0]))
    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=13)
    for i in range(13):
        f[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        f[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
    f["duration_seconds"] = float(librosa.get_duration(y=y, sr=TARGET_SR))
    return f
