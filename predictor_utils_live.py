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

MODEL_DIR = "data/training_data/ESP/saved_gb_model_no_leakage"
EMBED_DIR = "data/training_data/ESP/embeddings"
TRANSFORMER_PATH = "data/training_data/ESP/saved_model/ESP_2gpus_bs48_1e-05_layers6.txt.pkl"

SMILES_BERT = "DeepChem/ChemBERTa-77M-MTR"

MAX_SMILES_LEN = 256
MAX_PROT_LEN = 1018

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# LOAD PROJECT MODULES
# =========================================

def setup_project_path():
    """Set the working directory and make project training utilities importable."""
    os.chdir(PROJECT_DIR)

    if TRAINING_CODE_DIR not in sys.path:
        sys.path.append(TRAINING_CODE_DIR)


# =========================================
# LOAD SAVED EMBEDDINGS
# =========================================

def load_all_smiles_embeddings(embed_dir):
    smiles_all = {}
    smiles_path = os.path.join(embed_dir, "SMILES")

    for file in os.listdir(smiles_path):
        if file.endswith(".pkl"):
            with open(os.path.join(smiles_path, file), "rb") as f:
                smiles_all.update(pickle.load(f))

    return smiles_all


def load_all_protein_embeddings(embed_dir):
    protein_all = {}
    protein_path = os.path.join(embed_dir, "Protein")

    for file in os.listdir(protein_path):
        if file.endswith(".pt"):
            protein_all.update(
                torch.load(os.path.join(protein_path, file), map_location="cpu")
            )

    return protein_all


# =========================================
# LOAD FULL PREDICTOR
# =========================================

def load_predictor(load_live_models=True):
    """
    Load the saved XGBoost ensemble, transformer model, saved embeddings,
    and optionally the live embedding models ChemBERTa and ESM1b.

    load_live_models=True:
        Enables predictions for new SMILES/protein sequences not present in the saved dictionaries.
        This is slower and uses more memory.

    load_live_models=False:
        Faster loading, but predictions only work for inputs already present in saved embeddings.
    """
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

    print("Loading GB models...")

    gb_all = xgb.Booster()
    gb_all.load_model(os.path.join(MODEL_DIR, "gb_all.json"))

    gb_all_cls = xgb.Booster()
    gb_all_cls.load_model(os.path.join(MODEL_DIR, "gb_all_cls.json"))

    gb_cls = xgb.Booster()
    gb_cls.load_model(os.path.join(MODEL_DIR, "gb_cls.json"))

    with open(os.path.join(MODEL_DIR, "ensemble_weights.json"), "r") as f:
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

    state_dict = torch.load(TRANSFORMER_PATH, map_location=device)

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    transformer_model = model

    print("Loading saved embeddings...")

    smiles_embedding_dict = load_all_smiles_embeddings(EMBED_DIR)
    protein_embedding_dict = load_all_protein_embeddings(EMBED_DIR)

    if load_live_models:
        print("Loading ChemBERTa for live SMILES embeddings...")
        smiles_tokenizer = AutoTokenizer.from_pretrained(SMILES_BERT)
        smiles_bert = AutoModelForMaskedLM.from_pretrained(SMILES_BERT)
        smiles_bert = smiles_bert.to(device)
        smiles_bert.eval()

        print("Loading ESM1b for live protein embeddings...")
        esm_model, esm_alphabet = pretrained.load_model_and_alphabet("esm1b_t33_650M_UR50S")
        esm_model = esm_model.to(device)
        esm_model.eval()
        esm_batch_converter = esm_alphabet.get_batch_converter()
    else:
        smiles_tokenizer = None
        smiles_bert = None
        esm_model = None
        esm_alphabet = None
        esm_batch_converter = None

    print("Predictor loaded successfully.")
    print("Device:", device)
    print("SMILES embeddings:", len(smiles_embedding_dict))
    print("Protein embeddings:", len(protein_embedding_dict))
    print("Weights:", weights)
    print("Live embedding models loaded:", load_live_models)


# =========================================
# VALIDATION AND HELPER FUNCTIONS
# =========================================

def validate_predictor_loaded():
    if gb_all is None or gb_all_cls is None or gb_cls is None:
        raise RuntimeError("GB models are not loaded. Run load_predictor() first.")

    if weights is None:
        raise RuntimeError("Ensemble weights are not loaded. Run load_predictor() first.")

    if transformer_model is None:
        raise RuntimeError("Transformer model is not loaded. Run load_predictor() first.")

    if smiles_embedding_dict is None or protein_embedding_dict is None:
        raise RuntimeError("Embedding dictionaries are not loaded. Run load_predictor() first.")


def validate_input(smiles, protein_sequence):
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        raise ValueError("SMILES is empty or not a string.")

    if not isinstance(protein_sequence, str) or len(protein_sequence.strip()) == 0:
        raise ValueError("Protein sequence is empty or not a string.")

    smiles = smiles.strip()
    protein_sequence = protein_sequence.replace("\n", "").replace(" ", "").upper()

    allowed_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    unknown_chars = set(protein_sequence) - allowed_amino_acids

    if len(unknown_chars) > 0:
        print("Warning: protein sequence contains unusual characters:", unknown_chars)

    if len(protein_sequence) > MAX_PROT_LEN:
        print("Warning: protein sequence is longer than 1018 amino acids and will be truncated.")

    return smiles, protein_sequence


def pad_embedding(emb, max_len):
    if not torch.is_tensor(emb):
        emb = torch.tensor(emb)

    emb = emb.float()
    emb_len = emb.shape[0]
    dim = emb.shape[1]

    if emb_len > max_len:
        emb = emb[:max_len]
        emb_len = max_len

    padded = torch.zeros(max_len, dim)
    padded[:emb_len, :] = emb

    attn = torch.zeros(max_len)
    attn[:emb_len] = 1

    return padded, attn


def get_cls_representation(smiles_emb, protein_emb):
    smiles_padded, smiles_attn = pad_embedding(smiles_emb, MAX_SMILES_LEN)
    protein_padded, protein_attn = pad_embedding(protein_emb, MAX_PROT_LEN)

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
# LIVE EMBEDDING GENERATION
# =========================================

def generate_live_smiles_embedding(smiles):
    """
    Generate a ChemBERTa SMILES embedding for a new SMILES string.
    This follows the original project logic using AutoModelForMaskedLM and the 'logits' output.
    """
    if smiles_tokenizer is None or smiles_bert is None:
        raise RuntimeError(
            "ChemBERTa is not loaded. Run load_predictor(load_live_models=True) first."
        )

    tokens = smiles_tokenizer(
        smiles,
        max_length=500,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    tokens = {key: value.to(device) for key, value in tokens.items()}

    with torch.no_grad():
        smiles_emb = smiles_bert(**tokens)["logits"]

    return smiles_emb.squeeze(0).detach().cpu()


def generate_live_protein_embedding(protein_sequence):
    """
    Generate an ESM1b protein embedding for a new protein sequence.
    Uses representation layer 33 and truncates to the same max length as the original project.
    """
    if esm_model is None or esm_batch_converter is None:
        raise RuntimeError(
            "ESM1b is not loaded. Run load_predictor(load_live_models=True) first."
        )

    protein_sequence = protein_sequence[:MAX_PROT_LEN]

    data = [("protein_input", protein_sequence)]
    labels, strs, toks = esm_batch_converter(data)

    toks = toks[:, :1022].to(device)

    with torch.no_grad():
        out = esm_model(
            toks,
            repr_layers=[33],
            return_contacts=False
        )

    protein_emb = out["representations"][33][0, 1:len(protein_sequence) + 1]

    return protein_emb.detach().cpu()


def get_smiles_embedding(smiles, allow_live_embeddings=True, cache_live_embedding=True):
    if smiles in smiles_embedding_dict:
        return smiles_embedding_dict[smiles].squeeze(), "saved embedding"

    if not allow_live_embeddings:
        raise ValueError("SMILES not found in saved embeddings.")

    print("SMILES not found in saved embeddings. Generating live ChemBERTa embedding...")
    smiles_emb = generate_live_smiles_embedding(smiles)

    if cache_live_embedding:
        smiles_embedding_dict[smiles] = smiles_emb.unsqueeze(0)

    return smiles_emb, "live ChemBERTa embedding"


def get_protein_embedding(protein_sequence, allow_live_embeddings=True, cache_live_embedding=True):
    protein_sequence = protein_sequence[:MAX_PROT_LEN]

    if protein_sequence in protein_embedding_dict:
        return torch.from_numpy(protein_embedding_dict[protein_sequence]), "saved embedding"

    if not allow_live_embeddings:
        raise ValueError("Protein sequence not found in saved embeddings.")

    print("Protein sequence not found in saved embeddings. Generating live ESM1b embedding...")
    protein_emb = generate_live_protein_embedding(protein_sequence)

    if cache_live_embedding:
        protein_embedding_dict[protein_sequence] = protein_emb.numpy()

    return protein_emb, "live ESM1b embedding"


# =========================================
# PREDICTION FUNCTIONS
# =========================================

def predict_activity(
    smiles,
    protein_sequence,
    allow_live_embeddings=True,
    cache_live_embeddings=True
):
    """
    Predict activity for a SMILES + protein sequence pair.

    If embeddings are already saved, they are reused.
    If not, live embeddings can be generated when allow_live_embeddings=True.
    """
    validate_predictor_loaded()

    smiles, protein_sequence = validate_input(smiles, protein_sequence)
    protein_sequence = protein_sequence[:MAX_PROT_LEN]

    smiles_emb, smiles_source = get_smiles_embedding(
        smiles,
        allow_live_embeddings=allow_live_embeddings,
        cache_live_embedding=cache_live_embeddings
    )

    protein_emb, protein_source = get_protein_embedding(
        protein_sequence,
        allow_live_embeddings=allow_live_embeddings,
        cache_live_embedding=cache_live_embeddings
    )

    smiles_mean = smiles_emb.mean(0).detach().cpu().numpy()
    protein_mean = protein_emb.mean(0).detach().cpu().numpy()

    cls_repr = get_cls_representation(smiles_emb, protein_emb)

    X_all = np.concatenate([protein_mean, smiles_mean]).reshape(1, -1)
    X_all_cls = np.concatenate([protein_mean, smiles_mean, cls_repr]).reshape(1, -1)
    X_cls = cls_repr.reshape(1, -1)

    pred_all = gb_all.predict(xgb.DMatrix(X_all))[0]
    pred_all_cls = gb_all_cls.predict(xgb.DMatrix(X_all_cls))[0]
    pred_cls = gb_cls.predict(xgb.DMatrix(X_cls))[0]

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


def print_prediction_result(smiles, protein_sequence, result):
    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"SMILES: {smiles}")
    print(f"Protein sequence length: {len(protein_sequence)} amino acids")
    print("-" * 60)
    print(f"Final activity score: {result['final_score']:.4f}")
    print(f"Predicted class: {result['pred_class']}")
    print(f"Confidence: {result['confidence']}")
    print(f"SMILES embedding source: {result.get('smiles_embedding_source', 'unknown')}")
    print(f"Protein embedding source: {result.get('protein_embedding_source', 'unknown')}")
    print("-" * 60)
    print("Submodel scores:")
    print(f"  GB all      : {result['pred_all']:.4f}")
    print(f"  GB all + CLS: {result['pred_all_cls']:.4f}")
    print(f"  GB CLS only : {result['pred_cls']:.4f}")
    print("-" * 60)
    print("Ensemble weights:")
    print(f"  w_all_cls: {result['weights']['w_all_cls']}")
    print(f"  w_all    : {result['weights']['w_all']}")
    print(f"  w_cls    : {result['weights']['w_cls']}")
    print("=" * 60)


def run_single_prediction(
    smiles,
    protein_sequence,
    allow_live_embeddings=True,
    cache_live_embeddings=True
):
    smiles, protein_sequence = validate_input(smiles, protein_sequence)
    result = predict_activity(
        smiles,
        protein_sequence,
        allow_live_embeddings=allow_live_embeddings,
        cache_live_embeddings=cache_live_embeddings
    )
    print_prediction_result(smiles, protein_sequence, result)


def predict_batch(
    input_df,
    allow_live_embeddings=True,
    cache_live_embeddings=True
):
    results = []

    for idx, row in input_df.iterrows():
        try:
            result = predict_activity(
                row["SMILES"],
                row["Protein sequence"],
                allow_live_embeddings=allow_live_embeddings,
                cache_live_embeddings=cache_live_embeddings
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
                "smiles_embedding_source": "",
                "protein_embedding_source": "",
                "error": str(e)
            })

    return pd.DataFrame(results)
