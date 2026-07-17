# Notebooks

Each notebook documents one member's part of the preprocessing and modelling
work. They are the training / analysis records; the integrated, runnable system
is `main.py` at the repo root.

| Notebook | Author | What it does | Produces |
|---|---|---|---|
| `Formative2_data_clean_merge_product_recommendation_model.ipynb` | Emmanuel Mukasa Simiyu | Loads the two source spreadsheets, cleans them (nulls, duplicates, type fixes), merges social profiles with transactions on the shared customer key, runs EDA, engineers features, and trains the product-recommendation model. | `merged_dataset.csv`, `product_recommendation_model.joblib` |
| `IMAGE_COLLECTION_AND_PROCESSING.ipynb` | Hikma Hamza | Loads each member's face photos, detects/crops faces, applies augmentations (rotation, flip, grayscale, noise), extracts MobileNetV2 embeddings, and trains the face-recognition model. | `image_features.csv`, `face_recognition_model.pkl` |
| `Audio_Proccessing_Pipeline.ipynb` | Silver Jr Shalom Nshuti | Loads each member's voice clips, plots waveforms and spectrograms, applies augmentations (pitch shift, time stretch, noise), extracts MFCC + spectral features, and trains the voiceprint authorisation model. | `audio_features.csv`, `voice_model.pkl` |
| *(integration)* `../main.py` | Henry Christian Parfait Uhiriwe | Combines all three models into one command-line pipeline (face -> identity map -> product -> voice), including the unauthorised-attempt path. | the working CLI demo |


## Reproducibility

All models use `random_state=42`.
