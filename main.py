"""
main.py  --  Integrated Identity & Product Recommendation System (Person 4)
===========================================================================

Wires the three teammate models into ONE command-line pipeline, in the exact
order the assignment specifies:

    face image --> [Face Recognition]  --fail--> ACCESS DENIED
                        | pass (identified as a known member)
                        v
             member_id_map:  name --> customer_id
                        v
        merged_dataset row --> [Product Recommendation]  (computed, HELD back)
                        v
    voice clip --> [Voice Verification]  --fail--> ACCESS DENIED
                        | pass (approved)
                        v
                 REVEAL the held product recommendation

Note on the flow: the FACE gate authorises *running* the product model; the
VOICE gate authorises *revealing* the result. If voice fails, the product is
NEVER shown. This matches Task 6 ("voice -> approves & displays the prediction").

Usage
-----
  python main.py --face PATH --voice PATH      # run one full transaction
  python main.py --demo                        # authorised run + stranger run
  python main.py --unauthorized                # stranger attempt only
  python main.py --stub                        # force stub mode (no ML libs)

If any model/CSV file is missing, that stage automatically falls back to a
stub so you can test the wiring before every teammate file has landed.
"""

import os
# --- quiet the noisy-but-harmless logs BEFORE heavy libs load ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")     # silence TensorFlow info/warns
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")    # silence oneDNN notice
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import warnings
warnings.filterwarnings("ignore")                       # sklearn version + misc UserWarnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

import argparse
import sys
from pathlib import Path

import pandas as pd

import pipeline_utils as pu

# ---------------------------------------------------------------------------
# Paths (edit only if you move things)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"

PRODUCT_MODEL = MODELS_DIR / "product_recommendation_model.joblib"
FACE_MODEL = MODELS_DIR / "face_recognition_model.pkl"
VOICE_MODEL = MODELS_DIR / "voice_model.pkl"

MERGED_CSV = PROCESSED_DIR / "merged_dataset.csv"
ID_MAP_CSV = ROOT / "member_id_map.csv"

TEAM_MEMBERS = ["hikma", "silver", "christian", "emmanuel"]

# The face model was trained on filename-derived labels. Two members recorded
# their photos and audio under different names; the face model therefore still
# outputs the old photo labels. This map translates the model's raw output to
# the agreed canonical name so identity is consistent across the whole system.
#   photo label  ->  canonical name
LABEL_ALIASES = {"mukasa": "emmanuel", "shalom": "silver"}

FACE_THRESHOLD = 0.70   # from P2 notebook
VOICE_THRESHOLD = 0.65  # from P3 bundle (overridden by bundle value if present)

# Neutral fallback customer used only if a mapped customer_id is missing from
# the merged dataset (keeps the demo alive instead of crashing).
FALLBACK_CUSTOMER = {
    "purchase_amount": 200, "customer_rating": 4.0, "avg_engagement_score": 80,
    "avg_purchase_interest_score": 4.2, "n_social_records": 3, "purchase_month": 8,
    "purchase_dayofweek": 2, "is_weekend": 0, "sentiment_score": 1,
    "top_platform": "Instagram", "dominant_sentiment": "Positive",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_id_map():
    if ID_MAP_CSV.exists():
        m = pd.read_csv(ID_MAP_CSV)
        return dict(zip(m["first_name"].str.lower(), m["customer_id"]))
    return {}


def load_models(force_stub=False):
    """Return a dict describing which real models are available."""
    import joblib
    state = {"product": None, "face": None, "voice": None, "merged": None}
    if force_stub:
        return state

    try:
        if PRODUCT_MODEL.exists():
            state["product"] = joblib.load(PRODUCT_MODEL)
    except Exception as e:
        print(f"  [warn] could not load product model: {e}")

    try:
        if FACE_MODEL.exists():
            try:
                state["face"] = joblib.load(FACE_MODEL)
            except Exception:
                import pickle
                with open(FACE_MODEL, "rb") as fh:
                    state["face"] = pickle.load(fh)
    except Exception as e:
        print(f"  [warn] could not load face model: {e}")

    try:
        if VOICE_MODEL.exists():
            state["voice"] = joblib.load(VOICE_MODEL)
    except Exception as e:
        print(f"  [warn] could not load voice model: {e}")

    try:
        if MERGED_CSV.exists():
            state["merged"] = pd.read_csv(MERGED_CSV)
    except Exception as e:
        print(f"  [warn] could not load merged dataset: {e}")

    return state


# ---------------------------------------------------------------------------
# Stage 1 -- FACE
# ---------------------------------------------------------------------------
def _face_stub(face_path):
    who = Path(face_path).stem.split("_")[0].lower()
    who = LABEL_ALIASES.get(who, who)
    if who in TEAM_MEMBERS:
        return True, who, 0.99, "[stub]"
    return False, None, 0.10, "[stub] unknown face"


def face_gate(face_path, models):
    face_path = Path(face_path)
    if models["face"] is None:  # model file missing -> stub
        return _face_stub(face_path)

    # Real face path needs OpenCV (detect/crop) + TensorFlow (embed). If either
    # is missing or broken in this environment, fall back to the stub instead of
    # crashing, so the rest of the pipeline still runs live.
    try:
        import cv2  # noqa: F401
        if not hasattr(cv2, "CascadeClassifier"):
            raise ImportError("cv2 present but incomplete (no CascadeClassifier)")
        import tensorflow  # noqa: F401
    except Exception as e:
        print(f"  [info] face libraries unavailable ({e}); using face stub")
        return _face_stub(face_path)

    face = pu.extract_face(face_path)
    if face is None:
        return False, None, 0.0, "no face detected in image"
    emb = pu.create_embedding(face)
    X = pu.face_embedding_dataframe(emb)
    model = models["face"]
    raw_name = str(model.predict(X)[0]).lower()
    name = LABEL_ALIASES.get(raw_name, raw_name)   # translate to canonical name
    conf = float(max(model.predict_proba(X)[0]))
    if conf >= FACE_THRESHOLD and name in TEAM_MEMBERS:
        return True, name, conf, ""
    return False, name, conf, "below threshold / not a known member"


# ---------------------------------------------------------------------------
# Stage 2 -- map name -> customer features -> PRODUCT (held)
# ---------------------------------------------------------------------------
def product_step(member_name, id_map, models):
    customer_id = id_map.get(member_name)

    if models["product"] is None:  # ---- stub ----
        pick = ["Electronics", "Sports", "Books", "Clothing", "Groceries"]
        return pick[hash(member_name) % len(pick)], 0.42, customer_id, "[stub]"

    bundle = models["product"]
    pipeline = bundle["pipeline"]
    le = bundle["label_encoder"]
    needed = bundle["numeric_features"] + bundle["categorical_features"]

    features = None
    if customer_id is not None and models["merged"] is not None:
        rows = models["merged"][models["merged"]["customer_id"] == customer_id]
        if len(rows) > 0:
            features = rows.iloc[0][needed].to_dict()
    if features is None:
        features = {k: FALLBACK_CUSTOMER[k] for k in needed if k in FALLBACK_CUSTOMER}

    row = pd.DataFrame([features])
    enc = pipeline.predict(row)[0]
    proba = pipeline.predict_proba(row)[0]
    label = le.inverse_transform([enc])[0]
    return str(label), float(proba[enc]), customer_id, ""


# ---------------------------------------------------------------------------
# Stage 3 -- VOICE
# ---------------------------------------------------------------------------
def voice_gate(voice_path, models):
    voice_path = Path(voice_path)
    if models["voice"] is None:  # ---- stub ----
        who = voice_path.stem.split("_")[0].lower()
        if who in TEAM_MEMBERS:
            return True, 0.95, "[stub]"
        return False, 0.15, "[stub] unknown voice"

    bundle = models["voice"]
    model = bundle["model"]
    feat_cols = bundle["feature_columns"]
    thresh = float(bundle.get("threshold", VOICE_THRESHOLD))

    y = pu.load_audio(voice_path)
    feats = pu.extract_audio_features(y)
    X = pd.DataFrame([feats]).reindex(columns=feat_cols, fill_value=0.0)
    prob = float(model.predict_proba(X)[0][1])
    return (prob >= thresh), prob, ""


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_pipeline(face_path, voice_path, models, id_map, label="TRANSACTION"):
    line = "=" * 68
    print(f"\n{line}\n  {label}\n{line}")
    print(f"  face : {face_path}")
    print(f"  voice: {voice_path}\n")

    # Stage 1 -- face
    ok, name, conf, note = face_gate(face_path, models)
    print(f"[1] FACE RECOGNITION  ->  {'RECOGNISED' if ok else 'REJECTED'} "
          f"({name}, conf={conf:.2f}) {note}")
    if not ok:
        print("\n  ACCESS DENIED at face stage. Pipeline halted.\n")
        return

    # Stage 2 -- product (computed but withheld)
    product, p_conf, cid, note = product_step(name, id_map, models)
    print(f"[2] IDENTITY MAP      ->  {name} = customer_id {cid}")
    print(f"[3] PRODUCT MODEL     ->  computed & HELD (revealed only after voice) {note}")

    # Stage 3 -- voice
    ok, v_prob, note = voice_gate(voice_path, models)
    print(f"[4] VOICE VERIFICATION -> {'APPROVED' if ok else 'REJECTED'} "
          f"(prob={v_prob:.2f}) {note}")
    if not ok:
        print("\n  ACCESS DENIED at voice stage. Product recommendation NOT shown.\n")
        return

    # Reveal
    print(f"\n  ACCESS GRANTED for {name.title()}.")
    print(f"  >>> RECOMMENDED PRODUCT: {product}  (confidence {p_conf:.2f})\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Identity + product recommendation demo")
    ap.add_argument("--face")
    ap.add_argument("--voice")
    ap.add_argument("--demo", action="store_true", help="one authorised run + stranger run")
    ap.add_argument("--all", action="store_true",
                    help="run EVERY team member in turn, then a stranger (for the demo video)")
    ap.add_argument("--unauthorized", action="store_true", help="stranger attempt only")
    ap.add_argument("--stub", action="store_true", help="force stub mode")
    args = ap.parse_args()

    models = load_models(force_stub=args.stub)
    id_map = load_id_map()

    status = {k: ("real" if v is not None else "stub/missing") for k, v in models.items()}
    print("Model status:", status)
    print("ID map:", id_map or "(none loaded)")

    if args.all:
        # Every member: their smile photo + their approve clip -> expect ACCESS GRANTED
        for member in TEAM_MEMBERS:
            run_pipeline(f"data/raw/images/{member}_smile.jpeg",
                         f"data/raw/audio/{member}_approve.wav",
                         models, id_map, f"MEMBER: {member.upper()} (expect GRANTED)")
        # Stranger: unknown face -> expect ACCESS DENIED at the face gate
        run_pipeline("test_images/stranger.jpg",
                     "test_audio/stranger.wav",
                     models, id_map, "UNAUTHORISED ATTEMPT (stranger, expect DENIED)")
    elif args.demo:
        run_pipeline("data/raw/images/hikma_smile.jpeg",
                     "data/raw/audio/hikma_approve.wav",
                     models, id_map, "FULL TRANSACTION (authorised)")
        run_pipeline("test_images/stranger.jpg",
                     "test_audio/stranger.wav",
                     models, id_map, "UNAUTHORISED ATTEMPT (stranger)")
    elif args.unauthorized:
        run_pipeline("test_images/stranger.jpg",
                     "test_audio/stranger.wav",
                     models, id_map, "UNAUTHORISED ATTEMPT (stranger)")
    elif args.face and args.voice:
        run_pipeline(args.face, args.voice, models, id_map)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
