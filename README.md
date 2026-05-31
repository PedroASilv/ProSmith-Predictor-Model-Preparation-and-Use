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

Open predictor_utils_live.py en plak daarin de volgende code:

```bash



# Stap 10: Maak een Jupyter kernel aan
Je gaat nu een kernel aanmaken zodat de predictor vanuit Jupyter Notebook gebruikt kan worden. Voer deze code in:

```bash
python -m ipykernel install \
    --user \
    --name esp_predictor \
    --display-name "Python (esp_predictor)"
```

De naam van de kernel mag aangepast worden aan de gebruikte omgeving.

Start ter controle JupyterLab op en voer in:

```bash
jupyter lab
```


