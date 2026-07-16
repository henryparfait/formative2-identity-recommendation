# Identity Verification & Product Recommendation System

A command-line system that authenticates a user through **face recognition**
and **voice verification** before revealing a personalised **product
recommendation**. Built from three separately-developed models, integrated into
one pipeline.

## Pipeline order 

```
face image --> [Face Recognition] --fail--> ACCESS DENIED
                     | pass
                     v
          member_id_map: name --> customer_id
                     v
     merged_dataset --> [Product Recommendation]  (computed, HELD)
                     v
voice clip --> [Voice Verification] --fail--> ACCESS DENIED (product never shown)
                     | pass
                     v
              REVEAL product recommendation
```

The **face** gate authorises *running* the product model; the **voice** gate
authorises *revealing* the result. 
## Folder structure

```
.
├── main.py                     # the integrated CLI pipeline (run this)
├── pipeline_utils.py           # shared face/audio feature extractors
├── member_id_map.csv           # links each team member to a customer_id
├── requirements.txt
├── README.md
├── models/
│   ├── product_recommendation_model.joblib   # from Person 1
│   ├── face_recognition_model.pkl            # from Person 2
│   └── voice_model.pkl                       # from Person 3
├── data/
│   ├── raw/
│   │   ├── images/             # everyone's photos: name_neutral.jpg, name_smile.jpg, ...
│   │   └── audio/              # everyone's clips:  name_approve.wav, name_confirm.wav
│   └── processed/
│       ├── merged_dataset.csv  # from Person 1
│       ├── image_features.csv  # from Person 2
│       └── audio_features.csv  # from Person 3
├── notebooks/                  # the three source notebooks (one per person)
├── test_images/stranger.jpg    # unauthorised-attempt image
├── test_audio/stranger.wav     # unauthorised-attempt audio
└── outputs/                    # EDA / evaluation plots
```

## How to run

```bash
pip install -r requirements.txt

# one full transaction
python main.py --face data/raw/images/hikma_smile.jpeg --voice data/raw/audio/hikma_approve.wav

# authorised run followed by a stranger attempt
python main.py --demo

# stranger attempt only
python main.py --unauthorized
```



## Team members

Hikma, Shalom, Christian & Emmanuel. Each submitted 3 face images (neutral,
smile, surprised) and 2 audio phrases ("Yes, approve" / "Confirm transaction").

## Identity linkage (important, and stated honestly)

The face, voice and product models were built on three unrelated data sources:

- the **face** model predicts a *member name* (hikma, shalom, ...);
- the **voice** model predicts *authorised vs unauthorised* (it confirms the
  speaker is a group member, not *which* member);
- the **product** model needs a *customer row* from the merged transaction /
  social dataset, keyed by `customer_id`.

There is no natural key between a group member and a synthetic customer in the
provided spreadsheets. We therefore define an explicit, deliberate mapping in
`member_id_map.csv` that assigns each member a real `customer_id` present in the
merged dataset. This linkage is **assigned for demonstration purposes** — it
lets the end-to-end system run on real customer features. It does not claim any
discovered relationship between a person's face/voice and their purchasing
behaviour.

## Known limitations

- **Voice = authorisation, not identification.** The voice model verifies that
  the speaker is *an* authorised member (binary), not that they are the specific
  person the face identified. Per-speaker voice verification would need a
  multi-class voiceprint model.
- **Small data.** Product recommendation trains on 150 transactions across 5
  classes, so accuracy is modest by design; face/voice train on a handful of
  samples per member and rely heavily on augmentation.
