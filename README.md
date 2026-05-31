# ProSmith_Predictor_Model_Preparation_and_Use
Deze handleiding beschrijft stap voor stap hoe de ESP enzyme-substrate predictor wordt opgezet, getraind en gebruikt in Jupyter Notebook. Hou daarbij rekening met het feit dat dit gemaakt is op basis van de super computers van SRAM SURF die draaien op Linux.

# VEREISTEN
Python 3.8

Micromamba of Conda

Jupyter Notebook of JupyterLab

Toegang tot de ProSmith repository

# Stap 1: Navigeer naar de projectrepository
Ga naar de map waarin de ProSmith repository zich bevindt. Controleer of je in de juiste map staat met:

```bash
pwd
```

De output moet verwijzen naar de hoofdmap van de repository. Bijvoorbeeld: /data/storage_prj61_2/ProSmith

Controleer vervolgens of je repository correct geladen is:

```bash
ls
```

Je zou mappen moeten zien zoals:
```text
code/
data/
README.md
```

# Stap 2: Activeer de juiste Python-environment
Activeer de environment waarin alle benodigde packages voor ProSmith zijn geïnstalleerd.

Voorbeeld met Micromamba:
```bash
eval "$(micromamba shell hook --shell bash)" micromamba activate <environment_naam>
```

In dit project werd in <environment_naam> ProSmith ingevuld.

Controleer vervolgensd of de juiste environment actief is:
```bash
which python
```

De output moet verwijzen naar de geactiveerde environment 

# Stap 3: Controleer de trainingsdata
Controleer of de train-, validatie- en testbestanden aanwezig zijn:

```bash
ls data/training_data/ESP/train_val/
```

Je verwacht dan bestanden te zien zoals:
```text
ESP_train_df.csv
ESP_val_df.csv 
ESP_test_df.csv
```

# Stap 4: Genereer embeddings (Preprocessing)
Maak proteïne- en SMILES-embeddings die later gebruikt worden voor training.

```bash
python code/preprocessing/preprocessing.py \
    --train_val_path data/training_data/ESP/train_val \
    --outpath data/training_data/ESP/embeddings \
    --prot_emb_no 2000 \
    --smiles_emb_no 2000
```

# Stap 5: Model training
De embeddings van de preprocessing worden nu gecombineerd om het model te trainen.

Om de training te starten:
```bash
export USE_LIBUV=0

python code/training/training.py \
    --train_dir data/training_data/ESP/train_val/ESP_train_df.csv \
    --val_dir data/training_data/ESP/train_val/ESP_val_df.csv \
    --save_model_path data/training_data/ESP/saved_model \
    --embed_path data/training_data/ESP/embeddings \
    --pretrained_model data/training_data/BindingDB/saved_model/pretraining_IC50_6gpus_bs144_1.5e-05_layers6.txt.pkl \
    --learning_rate 1e-5 \
    --num_hidden_layers 6 \
    --batch_size 24 \
    --binary_task True \
    --log_name ESP \
    --num_train_epochs 100 \
    --port 29500
```

# Stap 6: Controleer of de training succesvol is afgerond
Het is belangrijk dat het model correct is opgeslagen voordat de Gradient Boosting modellen worden getraind. Dit kan snel gecontroleerd worden met:

```bash
find data/training_data/ESP/saved_model -type f
```

Je verwacht dan iets te zien, zoals data/training_data/ESP/saved_model/ESP_2gpus_bs48_1e-05_layers6.txt.pkl of een vergelijkbaar modelbestand.

Wanneer dit bestand aanwezig is, kan worden doorgegaan naar de training van de Gradient Boosting modellen.

# Stap 7: Train de Gradient Boosting modellen
Train de drie Gradient Boosting modellen die later gebruikt worden door de predictor:

```bash
python code/training/training_GB.py \
    --train_dir data/training_data/ESP/train_val/ESP_train_df.csv \
    --val_dir data/training_data/ESP/train_val/ESP_val_df.csv \
    --test_dir data/training_data/ESP/train_val/ESP_test_df.csv \
    --pretrained_model data/training_data/ESP/saved_model/ESP_2gpus_bs48_1e-05_layers6.txt.pkl \
    --embed_path data/training_data/ESP/embeddings \
    --save_pred_path data/training_data/ESP/saved_predictions \
    --save_gb_model_path data/training_data/ESP/saved_gb_model \
    --num_hidden_layers 6 \
    --num_iter 500 \
    --log_name ESP \
    --binary_task True
```

Als dit gelukt is moet je controleren of alles netjes is opgeslagen
Voer het onderstaande commando uit in de terminal:

```bash
ls data/training_data/ESP/saved_gb_model/
```
Je verwacht dan de volgende opgeslagen modellen te zien:

```text
gb_all.json
gb_all_cls.json
gb_cls.json
ensemble_weights.json
```

# Stap 8: Koppel predictions aan de testset
Tijdens de training worden de voorspellingen van het model opgeslagen in aparte bestanden. Deze voorspellingen bevatten de activiteitsscores die het model heeft berekend voor iedere combinatie in de testset.

In deze stap worden deze voorspellingen gekoppeld aan een kopie van de oorspronkelijke testset. Hierdoor ontstaat een overzichtelijk bestand waarin zowel de werkelijke labels (output) als de voorspelde scores (y_pred) aanwezig zijn. Dit bestand wordt later gebruikt voor validatie, prestatieanalyse en het vergelijken van nieuwe voorspellingen met de oorspronkelijke modelresultaten. 

Voer daarvoor de onderstaande code uit in een Jupyter Notebook:

```python
import numpy as np
import pandas as pd
from os.path import join

y_pred = np.load(
    join(
        "data/training_data/ESP/saved_predictions",
        "y_test_pred.npy"
    )
)

y_pred_ind = np.load(
    join(
        "data/training_data/ESP/saved_predictions",
        "test_indices.npy"
    )
)

test_df = pd.read_csv(
    "data/training_data/ESP/train_val/ESP_test_df.csv"
)

test_df["y_pred"] = np.nan

for k, ind in enumerate(y_pred_ind):
    test_df.loc[int(ind), "y_pred"] = y_pred[k]

test_df.to_csv(
    "data/training_data/ESP/saved_predictions/ESP_test_with_predictions.csv",
    index=False
)

print("Saved ESP_test_with_predictions.csv")
```

# Stap 9: Plaats het predictorbestand
Het is nu tijd om de basis voor de live predictions op te zetten. 

Maak binnen de map "code" een nieuwe map aan met de naam "predictor".

In deze map moet je vervolgens een tekst bestand aanmaken genaamd "predictor_utils_live" en dit opslaan als .py

De mappenstructuur zou er uiteindelijk als volgt uit moeten zien:

```text
code/
└── Predictor/
    └── predictor_utils_live.py
```
Nu je een path gemaakt hebt voor je predictorbestand is het belangrijk dat het ook informatie bevat die later gebruikt kan worden door het model.

Open predictor_utils_live.py en plaats daarin de volgende code:

```bash
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from transformers import AutoTokenizer, AutoModelForMaskedLM
from esm import pretrained


# =========================================
# GLOBAL SETTINGS
# =========================================

PROJECT_DIR = "/data/storage_prj61_2/ProSmith"
TRAINING_CODE_DIR = os.path.join(PROJECT_DIR, "code", "training")

MODEL_DIR = "data/training_data/ESP/saved_gb_model"
EMBED_DIR = "data/training_data/ESP/embeddings"
TRANSFORMER_PATH = "data/training_data/ESP/saved_model/ESP_2gpus_bs48_1e-05_layers6.txt.pkl"

MAX_SMILES_LEN = 256
MAX_PROT_LEN = 1018

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================
# GLOBAL MODEL OBJECTS
# =========================================

gb_all = None
gb_all_cls = None
gb_cls = None
weights = None

transformer_model = None

smiles_embedding_dict = None
protein_embedding_dict = None

smiles_tokenizer = None
smiles_bert = None

esm_model = None
esm_alphabet = None
esm_batch_converter = None


# =========================================
# PROJECT SETUP
# =========================================

def setup_project_path():
    os.chdir(PROJECT_DIR)

    if TRAINING_CODE_DIR not in sys.path:
        sys.path.append(TRAINING_CODE_DIR)


# =========================================
# LOAD SAVED EMBEDDINGS
# =========================================

def load_all_smiles_embeddings(embed_dir):
    smiles_all = {}

    smiles_path = os.path.join(
        embed_dir,
        "SMILES"
    )

    for file in os.listdir(smiles_path):
        if file.endswith(".pkl"):
            with open(
                os.path.join(smiles_path, file),
                "rb"
            ) as f:
                smiles_all.update(
                    pickle.load(f)
                )

    return smiles_all


def load_all_protein_embeddings(embed_dir):
    protein_all = {}

    protein_path = os.path.join(
        embed_dir,
        "Protein"
    )

    for file in os.listdir(protein_path):
        if file.endswith(".pt"):
            protein_all.update(
                torch.load(
                    os.path.join(protein_path, file),
                    map_location="cpu"
                )
            )

    return protein_all


# =========================================
# LOAD FULL PREDICTOR
# =========================================

def load_predictor(load_live_models=True):
    global gb_all
    global gb_all_cls
    global gb_cls
    global weights

    global transformer_model

    global smiles_embedding_dict
    global protein_embedding_dict

    global smiles_tokenizer
    global smiles_bert

    global esm_model
    global esm_alphabet
    global esm_batch_converter

    setup_project_path()

    from utils.modules import MM_TN, MM_TNConfig

    print("Loading Gradient Boosting models...")

    gb_all = xgb.Booster()
    gb_all.load_model(
        os.path.join(MODEL_DIR, "gb_all.json")
    )

    gb_all_cls = xgb.Booster()
    gb_all_cls.load_model(
        os.path.join(MODEL_DIR, "gb_all_cls.json")
    )

    gb_cls = xgb.Booster()
    gb_cls.load_model(
        os.path.join(MODEL_DIR, "gb_cls.json")
    )

    with open(
        os.path.join(MODEL_DIR, "ensemble_weights.json"),
        "r"
    ) as f:
        weights = json.load(f)

    print("Loading transformer model...")

    config = MM_TNConfig.from_dict({
        "s_hidden_size": 600,
        "p_hidden_size": 1280,
        "hidden_size": 768,
        "max_seq_len": 1276,
        "num_hidden_layers": 6,
        "binary_task": True
    })

    model = MM_TN(config)

    state_dict = torch.load(
        TRANSFORMER_PATH,
        map_location=device
    )

    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(
        new_state_dict,
        strict=False
    )

    model = model.to(device)
    model.eval()

    transformer_model = model

    print("Loading saved embeddings...")

    smiles_embedding_dict = load_all_smiles_embeddings(
        EMBED_DIR
    )

    protein_embedding_dict = load_all_protein_embeddings(
        EMBED_DIR
    )

    if load_live_models:
        print("Loading ChemBERTa for live SMILES embeddings...")

        SMILES_BERT = "DeepChem/ChemBERTa-77M-MTR"

        smiles_tokenizer = AutoTokenizer.from_pretrained(
            SMILES_BERT
        )

        smiles_bert = AutoModelForMaskedLM.from_pretrained(
            SMILES_BERT
        )

        smiles_bert = smiles_bert.to(device)
        smiles_bert.eval()

        print("Loading ESM1b for live protein embeddings...")

        esm_model, esm_alphabet = pretrained.load_model_and_alphabet(
            "esm1b_t33_650M_UR50S"
        )

        esm_model = esm_model.to(device)
        esm_model.eval()

        esm_batch_converter = esm_alphabet.get_batch_converter()

    print("Predictor loaded successfully.")
    print("Device:", device)
    print("SMILES embeddings:", len(smiles_embedding_dict))
    print("Protein embeddings:", len(protein_embedding_dict))
    print("Weights:", weights)


# =========================================
# VALIDATION HELPERS
# =========================================

def validate_predictor_loaded():
    if gb_all is None or gb_all_cls is None or gb_cls is None:
        raise RuntimeError(
            "Gradient Boosting models are not loaded. "
            "Run load_predictor() first."
        )

    if weights is None:
        raise RuntimeError(
            "Ensemble weights are not loaded. "
            "Run load_predictor() first."
        )

    if transformer_model is None:
        raise RuntimeError(
            "Transformer model is not loaded. "
            "Run load_predictor() first."
        )

    if smiles_embedding_dict is None or protein_embedding_dict is None:
        raise RuntimeError(
            "Embedding dictionaries are not loaded. "
            "Run load_predictor() first."
        )


def validate_input(smiles, protein_sequence):
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        raise ValueError(
            "SMILES is empty or not a string."
        )

    if not isinstance(protein_sequence, str) or len(protein_sequence.strip()) == 0:
        raise ValueError(
            "Protein sequence is empty or not a string."
        )

    smiles = smiles.strip()

    protein_sequence = (
        protein_sequence
        .replace("\n", "")
        .replace(" ", "")
        .upper()
    )

    allowed_amino_acids = set(
        "ACDEFGHIKLMNPQRSTVWY"
    )

    unknown_chars = set(protein_sequence) - allowed_amino_acids

    if len(unknown_chars) > 0:
        print(
            "Warning: protein sequence contains unusual characters:",
            unknown_chars
        )

    if len(protein_sequence) > MAX_PROT_LEN:
        print(
            "Warning: protein sequence is longer than 1018 amino acids "
            "and will be truncated."
        )

    return smiles, protein_sequence


# =========================================
# EMBEDDING HELPERS
# =========================================

def pad_embedding(emb, max_len):
    if not torch.is_tensor(emb):
        emb = torch.tensor(emb)

    emb = emb.float()

    emb_len = emb.shape[0]
    dim = emb.shape[1]

    if emb_len > max_len:
        emb = emb[:max_len]
        emb_len = max_len

    padded = torch.zeros(
        max_len,
        dim
    )

    padded[:emb_len, :] = emb

    attn = torch.zeros(
        max_len
    )

    attn[:emb_len] = 1

    return padded, attn


def generate_live_smiles_embedding(smiles):
    if smiles_tokenizer is None or smiles_bert is None:
        raise RuntimeError(
            "ChemBERTa is not loaded. "
            "Run load_predictor(load_live_models=True)."
        )

    tokens = smiles_tokenizer(
        smiles,
        max_length=500,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    tokens = {
        key: value.to(device)
        for key, value in tokens.items()
    }

    with torch.no_grad():
        smiles_emb = smiles_bert(
            **tokens
        )["logits"]

    return smiles_emb.squeeze(0).detach().cpu()


def generate_live_protein_embedding(protein_sequence):
    if esm_model is None or esm_batch_converter is None:
        raise RuntimeError(
            "ESM1b is not loaded. "
            "Run load_predictor(load_live_models=True)."
        )

    protein_sequence = protein_sequence[:MAX_PROT_LEN]

    data = [
        ("protein_input", protein_sequence)
    ]

    labels, strs, toks = esm_batch_converter(
        data
    )

    toks = toks[:, :1022].to(device)

    with torch.no_grad():
        out = esm_model(
            toks,
            repr_layers=[33],
            return_contacts=False
        )

    protein_emb = out["representations"][33][
        0,
        1:len(protein_sequence) + 1
    ]

    return protein_emb.detach().cpu()


def get_cls_representation(smiles_emb, protein_emb):
    smiles_padded, smiles_attn = pad_embedding(
        smiles_emb,
        MAX_SMILES_LEN
    )

    protein_padded, protein_attn = pad_embedding(
        protein_emb,
        MAX_PROT_LEN
    )

    smiles_padded = smiles_padded.unsqueeze(0).to(device)
    protein_padded = protein_padded.unsqueeze(0).to(device)

    smiles_attn = smiles_attn.unsqueeze(0).to(device)
    protein_attn = protein_attn.unsqueeze(0).to(device)

    gpu = 0 if device.type == "cuda" else None

    with torch.no_grad():
        _, cls_repr = transformer_model(
            smiles_emb=smiles_padded,
            smiles_attn=smiles_attn,
            protein_emb=protein_padded,
            protein_attn=protein_attn,
            device=device,
            gpu=gpu,
            get_repr=True
        )

    return cls_repr[0].detach().cpu().numpy()


# =========================================
# CLASSIFICATION HELPERS
# =========================================

def classify_score(score):
    if score >= 0.8:
        return "active / 1"

    elif score <= 0.2:
        return "inactive / 0"

    else:
        return "uncertain / onbeslist"


def confidence_label(score):
    if score >= 0.9:
        return "very high confidence active"

    elif score >= 0.8:
        return "high confidence active"

    elif score >= 0.6:
        return "weakly active / uncertain"

    elif score > 0.4:
        return "uncertain"

    elif score > 0.2:
        return "weakly inactive / uncertain"

    elif score > 0.1:
        return "high confidence inactive"

    else:
        return "very high confidence inactive"


# =========================================
# PREDICTION FUNCTIONS
# =========================================

def predict_activity(
    smiles,
    protein_sequence,
    allow_live_embeddings=True
):
    validate_predictor_loaded()

    smiles, protein_sequence = validate_input(
        smiles,
        protein_sequence
    )

    protein_sequence = protein_sequence[:MAX_PROT_LEN]

    if smiles in smiles_embedding_dict:
        smiles_emb = smiles_embedding_dict[
            smiles
        ].squeeze()

        smiles_source = "saved embedding"

    else:
        if not allow_live_embeddings:
            raise ValueError(
                "SMILES not found in saved embeddings."
            )

        print(
            "SMILES not found in saved embeddings. "
            "Generating live ChemBERTa embedding..."
        )

        smiles_emb = generate_live_smiles_embedding(
            smiles
        )

        smiles_source = "live ChemBERTa embedding"

    if protein_sequence in protein_embedding_dict:
        protein_emb = torch.from_numpy(
            protein_embedding_dict[
                protein_sequence
            ]
        )

        protein_source = "saved embedding"

    else:
        if not allow_live_embeddings:
            raise ValueError(
                "Protein sequence not found in saved embeddings."
            )

        print(
            "Protein sequence not found in saved embeddings. "
            "Generating live ESM1b embedding..."
        )

        protein_emb = generate_live_protein_embedding(
            protein_sequence
        )

        protein_source = "live ESM1b embedding"

    smiles_mean = (
        smiles_emb
        .mean(0)
        .detach()
        .cpu()
        .numpy()
    )

    protein_mean = (
        protein_emb
        .mean(0)
        .detach()
        .cpu()
        .numpy()
    )

    cls_repr = get_cls_representation(
        smiles_emb,
        protein_emb
    )

    X_all = np.concatenate([
        protein_mean,
        smiles_mean
    ]).reshape(1, -1)

    X_all_cls = np.concatenate([
        protein_mean,
        smiles_mean,
        cls_repr
    ]).reshape(1, -1)

    X_cls = cls_repr.reshape(1, -1)

    pred_all = gb_all.predict(
        xgb.DMatrix(X_all)
    )[0]

    pred_all_cls = gb_all_cls.predict(
        xgb.DMatrix(X_all_cls)
    )[0]

    pred_cls = gb_cls.predict(
        xgb.DMatrix(X_cls)
    )[0]

    final_score = (
        weights["w_all_cls"] * pred_all_cls +
        weights["w_all"] * pred_all +
        weights["w_cls"] * pred_cls
    )

    return {
        "final_score": float(final_score),
        "pred_class": classify_score(final_score),
        "confidence": confidence_label(final_score),

        "pred_all": float(pred_all),
        "pred_all_cls": float(pred_all_cls),
        "pred_cls": float(pred_cls),

        "weights": weights,

        "smiles_embedding_source": smiles_source,
        "protein_embedding_source": protein_source
    }


def print_prediction_result(
    smiles,
    protein_sequence,
    result
):
    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(f"SMILES: {smiles}")
    print(
        f"Protein sequence length: "
        f"{len(protein_sequence)} amino acids"
    )

    print("-" * 60)

    print(
        f"Final activity score: "
        f"{result['final_score']:.4f}"
    )

    print(
        f"Predicted class: "
        f"{result['pred_class']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']}"
    )

    print(
        f"SMILES embedding source: "
        f"{result.get('smiles_embedding_source', 'unknown')}"
    )

    print(
        f"Protein embedding source: "
        f"{result.get('protein_embedding_source', 'unknown')}"
    )

    print("-" * 60)

    print("Submodel scores:")

    print(
        f"  GB all      : "
        f"{result['pred_all']:.4f}"
    )

    print(
        f"  GB all + CLS: "
        f"{result['pred_all_cls']:.4f}"
    )

    print(
        f"  GB CLS only : "
        f"{result['pred_cls']:.4f}"
    )

    print("-" * 60)

    print("Ensemble weights:")

    print(
        f"  w_all_cls: "
        f"{result['weights']['w_all_cls']}"
    )

    print(
        f"  w_all    : "
        f"{result['weights']['w_all']}"
    )

    print(
        f"  w_cls    : "
        f"{result['weights']['w_cls']}"
    )

    print("=" * 60)


def run_single_prediction(
    smiles,
    protein_sequence
):
    smiles, protein_sequence = validate_input(
        smiles,
        protein_sequence
    )

    result = predict_activity(
        smiles,
        protein_sequence
    )

    print_prediction_result(
        smiles,
        protein_sequence,
        result
    )


def predict_batch(input_df):
    results = []

    for idx, row in input_df.iterrows():
        try:
            result = predict_activity(
                row["SMILES"],
                row["Protein sequence"]
            )

            results.append({
                "index": idx,
                "SMILES": row["SMILES"],
                "Protein sequence": row["Protein sequence"],

                "final_score": result["final_score"],
                "pred_class": result["pred_class"],
                "confidence": result["confidence"],

                "pred_all": result["pred_all"],
                "pred_all_cls": result["pred_all_cls"],
                "pred_cls": result["pred_cls"],

                "smiles_embedding_source": result["smiles_embedding_source"],
                "protein_embedding_source": result["protein_embedding_source"],

                "error": ""
            })

        except Exception as e:
            results.append({
                "index": idx,
                "SMILES": row.get("SMILES", ""),
                "Protein sequence": row.get("Protein sequence", ""),

                "final_score": np.nan,
                "pred_class": "error",
                "confidence": "error",

                "pred_all": np.nan,
                "pred_all_cls": np.nan,
                "pred_cls": np.nan,

                "smiles_embedding_source": "error",
                "protein_embedding_source": "error",

                "error": str(e)
            })

    return pd.DataFrame(results)
```


# Stap 10: Maak een Jupyter kernel aan
Je gaat nu een kernel aanmaken zodat de predictor vanuit Jupyter Notebook gebruikt kan worden. Voer deze code in:

```bash
python -m ipykernel install \
    --user \
    --name esp_predictor \
    --display-name "Python (esp_predictor)"
```

De naam van de kernel mag aangepast worden aan de gebruikte omgeving.

# Stap 11: Open een nieuwe Jupyter Notebook
Maak een nieuwe notebook aan waarin de predictor geladen en getest kan worden.

Start JupyterLab:

```bash
jupyter lab
```

Maak vervolgens een nieuwe notebook aan en selecteer de kernel die in stap 10 is aangemaakt.

Bijvoorbeeld:

```text
Python (esp_predictor)
```

of

```text
Python (prosmith)
```

afhankelijk van de gekozen naam.

# Stap 12: Laad de predictor
Importeer de predictor en laad alle benodigde modellen, embeddings en configuratiebestanden.

Voer onderstaande code uit in de notebook:

```python
import sys

sys.path.append(
    "/pad/naar/ProSmith/code/Predictor"
)

import predictor_utils_live as pred

pred.load_predictor(
    load_live_models=True
)
```

Na een tijdje zou de predictor volledig geladen moeten zijn.

Afhankelijk van de implementatie verschijnen meldingen over:

* ChemBERTa
* ESM1b
* Transformer model
* Gradient Boosting modellen
* Ensemble gewichten

De notebook mag pas verder gebruikt worden wanneer alle onderdelen succesvol geladen zijn.

# Stap 13: Controleer of de predictor correct geladen is
Verifieer eerst dat alle benodigde objecten beschikbaar zijn voordat voorspellingen worden uitgevoerd. Gebruik hiervoor de command:

```python
print(type(pred.smiles_embedding_dict))
print(type(pred.protein_embedding_dict))
```

Je verwacht dan iets te zien zoals:

```text
<class 'dict'>
<class 'dict'>
```

Controleer eventueel ook hoeveel embeddings geladen zijn:

```python
print(len(pred.smiles_embedding_dict))
print(len(pred.protein_embedding_dict))
```

Hier zouden positieve aantallen moeten verschijnen.

# Stap 14: Voer de eerste voorspelling uit
Test of de volledige pipeline werkt door een enzym-substraatcombinatie te voorspellen. Voer de command uit:

```python
SMILES = "CCO"

PROTEIN_SEQUENCE = """
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANN
""".replace("\n", "").replace(" ", "")

pred.run_single_prediction(
    SMILES,
    PROTEIN_SEQUENCE
)
```

De exacte score zal verschillen, maar de output zou vergelijkbaar moeten zijn met:

```text
PREDICTION RESULT

Final activity score: 0.XXX
Predicted class: ...

Confidence: ...

Submodel scores:
GB all ...
GB all + CLS ...
GB CLS only ...
```

Wanneer deze output verschijnt is de predictor succesvol geladen en functioneert de volledige pipeline.


