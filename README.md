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
## Stap 7a: Gradient Boosting modellen opslaan
De standaard workflow genereert voorspellingen, maar voor een standalone predictor moeten de getrainde Gradient Boosting modellen ook worden opgeslagen.
Hierdoor kunnen voorspellingen later worden uitgevoerd zonder de volledige training opnieuw te draaien.

Open
```text
code/training/training_GB.py
```
Voeg de onderstaande code toe:
```text
bst_all_test.save_model(
    os.path.join(
        args.save_gb_model_path,
        "gb_all.json"
    )
)

bst_all_cls_test.save_model(
    os.path.join(
        args.save_gb_model_path,
        "gb_all_cls.json"
    )
)

bst_cls_test.save_model(
    os.path.join(
        args.save_gb_model_path,
        "gb_cls.json"
    )
)

weights = {
    "w_all_cls": float(best_i),
    "w_all": float(best_j),
    "w_cls": float(best_k)
}

with open(
    os.path.join(
        args.save_gb_model_path,
        "ensemble_weights.json"
    ),
    "w"
) as f:
    json.dump(
        weights,
        f,
        indent=4
    )
```

Voer als controle uit:
```bash
ls data/training_data/ESP/saved_gb_model/
```
En je verwacht iets te zien zoals:
```text
gb_all.json
gb_all_cls.json
gb_cls.json
ensemble_weights.json
```
## Stap 7b: Test predictions koppelen aan de originele testset controle
Tijdens de oorspronkelijke training worden alleen de voorspellingen opgeslagen.
Om deze voorspellingen later terug te koppelen aan de originele testset worden ook de originele testindices opgeslagen. Hierdoor kan iedere voorspelling weer aan de juiste rij uit de testset gekoppeld worden.

Controleer of de onderstaande regels aanwezig zijn:
```text
np.save(
    join(
        args.save_pred_path,
        "y_test_pred.npy"
    ),
    y_test_pred
)

np.save(
    join(
        args.save_pred_path,
        "test_indices.npy"
    ),
    np.array(test_indices)
)
```
Hierdoor kun je later y_test_pred.npy en test_indices.npy combineren tot ESP_test_with_predictions.csv

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

# Stap 15: Voorspellen op bestaande testdata
Controleer nu of de predictor correct werkt op voorbeelden uit de testset.

Deze stap gebruikt bestaande enzym-substraatcombinaties uit de testdata. Hierdoor kan gecontroleerd worden of de predictor correct functioneert op data die al eerder door de pipeline is verwerkt.

Laad eerst de testset:

```python
import pandas as pd

pred_df = pd.read_csv(
    "data/training_data/ESP/saved_predictions/ESP_test_with_predictions.csv"
)
```

Selecteer vervolgens één voorbeeld uit de dataset:

```python
row = pred_df.iloc[0]

SMILES = row["SMILES"]

PROTEIN_SEQUENCE = row["Protein sequence"]

pred.run_single_prediction(
    SMILES,
    PROTEIN_SEQUENCE
)
```

De predictor zou een activiteitsscore moeten retourneren samen met:

```text
Final activity score
Predicted class
Confidence
Submodel scores
```

Wanneer deze voorspelling succesvol wordt uitgevoerd, functioneert de predictor correct op bestaande testdata.

---

# Stap 16: Voer batch predictions uit
Nu ga je kijken of er ook accuraat meerdere voorspellingen tegelijk uitgevoerd kunnen worden.

Dit is nuttig wanneer meerdere enzym-substraatcombinaties tegelijkertijd geëvalueerd moeten worden.

Selecteer een willekeurige subset uit de testdata:

```python
input_df = pred_df.sample(
    n=10,
    random_state=42
)
```

Voer vervolgens batch prediction uit:

```python
results = pred.predict_batch(
    input_df
)

results
```

Sla de resultaten op:

```python
results.to_csv(
    "batch_prediction_results.csv",
    index=False
)
```

Controleer of het bestand succesvol is aangemaakt:

```python
import os

print(
    os.path.exists(
        "batch_prediction_results.csv"
    )
)
```

Je verwacht dan terug te krijgen:

```text
True
```

Je kunt nu het opgeslagen bestand openen in excel en kijken naar de resultaten. Desnoods kun je ook statistiek op deze datasets uitvoeren.

# Stap 17: Voer een live prediction uit
Nu zou je in theorie een voorspelling uit kunnen voeren op een nieuwe enzym-substraatcombinatie. De success hiervan is extreem afhankelijk van de informatie die het model verkregen heeft via de training. Het is dus niet raar als je lage waardes hebt, omdat dit vooral betekent dat het model zulke combinaties nog niet herkent.

Wanneer de ingevoerde SMILES of aminozuursequentie niet aanwezig is in de opgeslagen embeddings, zal de predictor automatisch nieuwe embeddings genereren met:

```text
SMILES → ChemBERTa

Proteïne → ESM1b
```

Vervang onderstaande placeholders door een eigen SMILES en aminozuursequentie uit literatuur:

```python
SMILES = "<VOER_HIER_EEN_SMILES_IN>"

PROTEIN_SEQUENCE = """
<VOER_HIER_EEN_AMINOZUURSEQUENTIE_IN>
""".replace("\n", "").replace(" ", "")

pred.run_single_prediction(
    SMILES,
    PROTEIN_SEQUENCE
)
```

Wanneer de combinatie niet aanwezig is in de opgeslagen embeddings kan de output onder andere bevatten:

```text
SMILES embedding source: live ChemBERTa embedding

Protein embedding source: live ESM1b embedding
```

Dit bevestigt dat de live embedding pipeline correct functioneert.



# Stap 18: Minimale eindgebruikerscode
Alles is nu opgezet en klaar voor gebruik!

Onderstaande code bevat de minimale hoeveelheid code die een gebruiker nodig heeft om een voorspelling uit te voeren nadat de predictor volledig is opgezet.

Vervang de placeholders door een eigen SMILES en aminozuursequentie:

```python
import sys

sys.path.append(
    "/pad/naar/ProSmith/code/Predictor"
)

import predictor_utils_live as pred

pred.load_predictor(
    load_live_models=True
)

SMILES = "<VOER_HIER_EEN_SMILES_IN>"

PROTEIN_SEQUENCE = """
<VOER_HIER_EEN_AMINOZUURSEQUENTIE_IN>
""".replace("\n", "").replace(" ", "")

pred.run_single_prediction(
    SMILES,
    PROTEIN_SEQUENCE
)
```

Na het uitvoeren van deze code retourneert de predictor een activiteitsscore, classificatie en betrouwbaarheidsinschatting voor de ingevoerde enzym-substraatcombinatie.

# Bijlage A – Data leakage fix in `training_GB.py`
In de oorspronkelijke Gradient Boosting workflow werd de validatieset deels opnieuw gebruikt bij het trainen van de modellen die uiteindelijk op de testset werden geëvalueerd.
Dat is ongewenst, omdat de validatieset dan niet meer uitsluitend wordt gebruikt voor modelselectie. Hierdoor kan de uiteindelijke testprestatie te optimistisch worden.

De oplossing is dat de validatieset alleen gebruikt wordt voor:

- hyperparameterselectie
- ensemble weight selectie

en niet opnieuw wordt toegevoegd aan de trainingdata voor de finale testset-predictions.

## Stap 1: Open `training_GB.py`

Open het bestand:

```text
code/training/training_GB.py
```
## Stap 2: Pas het eerste Gradient Boosting model aan (ESM1b+ChemBERTa2)
Zoek deze code:
```text
bst_all_test, y_test_pred_all = get_predictions(
    param = trials.argmin,
    dM_train = dtrain_val,
    dM_val = dtest
)
```
Vervang dit door:
```text
bst_all_test, y_test_pred_all = get_predictions(
    param = trials.argmin,
    dM_train = dtrain,
    dM_val = dtest
)
```
## Stap 3: Pas het tweede Gradient Boosting model aan (ESM1b+ChemBERTa2+cls-token)
Zoek deze code:
```text
bst_all_cls_test, y_test_pred_all_cls = get_predictions(
    param = trials.argmin,
    dM_train = dtrain_val_all_cls,
    dM_val = dtest_all_cls
)
```
Vervang dit door:
```text
bst_all_cls_test, y_test_pred_all_cls = get_predictions(
    param = trials.argmin,
    dM_train = dtrain_all_cls,
    dM_val = dtest_all_cls
)
```
## Stap 4: Pas het derde Gradient Boosting model aan (cls-token)
Zoek deze code:
```text
bst_cls_test, y_test_pred_cls = get_predictions(
    param = trials.argmin,
    dM_train = dtrain_val_cls,
    dM_val = dtest_cls
)
```
Vervang dit door:
```text
bst_cls_test, y_test_pred_cls = get_predictions(
    param = trials.argmin,
    dM_train = dtrain_cls,
    dM_val = dtest_cls
)
```
## Stap 5: Controleer of de oude dtrain_val varianten niet meer worden gebruikt
De volgende objecten mogen nog bestaan:
```text
dtrain_val
dtrain_val_all_cls
dtrain_val_cls
```
Maar ze mogen niet meer gebruikt worden bij:
```text
bst_all_test
bst_all_cls_test
bst_cls_test
```
De testset-predictions moeten dus uitsluitend deze train-objecten gebruiken:
```text
dtrain
dtrain_all_cls
dtrain_cls
```
## Stap 6: Run de leakage-vrije Gradient Boosting training
Gebruik aparte outputmappen zodat de oude en leakage-vrije resultaten niet door elkaar lopen:
```bash
python code/training/training_GB.py \
    --train_dir data/training_data/ESP/train_val/ESP_train_df.csv \
    --val_dir data/training_data/ESP/train_val/ESP_val_df.csv \
    --test_dir data/training_data/ESP/train_val/ESP_test_df.csv \
    --pretrained_model data/training_data/ESP/saved_model/ESP_2gpus_bs48_1e-05_layers6.txt.pkl \
    --embed_path data/training_data/ESP/embeddings \
    --save_pred_path data/training_data/ESP/saved_predictions_no_leakage \
    --save_gb_model_path data/training_data/ESP/saved_gb_model_no_leakage \
    --num_hidden_layers 6 \
    --num_iter 500 \
    --log_name ESP_no_leakage \
    --binary_task True
```
## Stap 7: Controleer de output
Controleer of de leakage-vrije Gradient Boosting modellen zijn opgeslagen:
```bash
ls data/training_data/ESP/saved_gb_model_no_leakage/
```
Verwachte output:
```text
gb_all.json
gb_all_cls.json
gb_cls.json
ensemble_weights.json
```
Controleer ook of de predictions zijn opgeslagen:
```bash
ls data/training_data/ESP/saved_predictions_no_leakage/
```
Verwachte output:
```text
y_test_pred.npy
test_indices.npy
```
## Stap 8: Gebruik de leakage-vrije modellen in de predictor
Open:
```text
code/Predictor/predictor_utils_live.py
```
Zoek:
```text
MODEL_DIR = "data/training_data/ESP/saved_gb_model"
```
Vervang dit door:
```text
MODEL_DIR = "data/training_data/ESP/saved_gb_model_no_leakage"
```
Hierdoor gebruikt de live predictor de leakage-vrije Gradient Boosting modellen.
