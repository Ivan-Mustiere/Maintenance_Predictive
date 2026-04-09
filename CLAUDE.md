# CLAUDE.md – Projet Maintenance Prédictive (Valve Condition)

## Vue d'ensemble du projet

Tu es Data Scientist / ML Engineer dans une entreprise industrielle.
L'objectif est de construire un système de **maintenance prédictive** pour prédire si la condition de la valve d'un cycle de production hydraulique est **optimale (100%)** ou **non optimale**.

- **Type de problème** : Classification binaire supervisée
- **Dataset source** : UCI – Condition Monitoring of Hydraulic Systems
  → https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems
- **Données fournies** : `data_subset.zip` contenant 3 fichiers

---

## Structure du projet attendue

```
projet-maintenance-predictive/
├── CLAUDE.md
├── README.md
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci_cd.yml
├── data/
│   ├── raw/                  # PS2.txt, FS1.txt, profile.txt (non versionnés via DVC)
│   └── processed/            # Features engineered, prêtes au ML
├── notebooks/
│   └── exploration.ipynb     # EDA + démarche documentée
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py      # Chargement des fichiers bruts
│   │   └── preprocess.py     # Feature engineering + split train/test
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py          # Entraînement du modèle
│   │   ├── evaluate.py       # Évaluation sur le test set final
│   │   └── predict.py        # Prédiction à partir d'un numéro de cycle
│   └── utils/
│       ├── __init__.py
│       └── logger.py
├── api/
│   ├── __init__.py
│   └── app.py                # Application web FastAPI
├── tests/
│   ├── __init__.py
│   ├── test_load_data.py
│   ├── test_preprocess.py
│   ├── test_predict.py
│   └── test_api.py
├── models/                   # Modèles sérialisés (.pkl ou .joblib), versionnés via DVC
│   └── model_v1.pkl
├── monitoring/
│   └── grafana/              # Dashboards Grafana (JSON)
└── dvc.yaml                  # Pipeline DVC pour versioning data + modèle
```

---

## Données

### Fichiers fournis (dans `data/raw/`)

| Fichier       | Description                                      | Fréquence d'échantillonnage |
|---------------|--------------------------------------------------|-----------------------------|
| `PS2.txt`     | Pression (bar)                                   | 100 Hz → 6000 valeurs/cycle |
| `FS1.txt`     | Débit volumique (L/min)                          | 10 Hz → 600 valeurs/cycle   |
| `profile.txt` | Variables cibles dont `valve condition` (col. 4) | 1 valeur/cycle              |

### Format des fichiers
- Séparateur : tabulation (`\t`)
- Pas d'en-tête
- Chaque ligne = 1 cycle de production

### Variable cible : `valve condition` (colonne index 3 dans `profile.txt`)
- `100` → condition optimale → **classe 1**
- Toute autre valeur (`90`, `80`, `73`) → non optimale → **classe 0**

### Split train/test (contrainte stricte du projet)
- **Train** : les **2000 premiers cycles** (index 0 à 1999)
- **Test final** : les **cycles restants** (index 2000 jusqu'à la fin)
- ⚠️ Ne jamais utiliser le test set pendant l'entraînement ou la validation croisée

---

## Feature Engineering

Chaque cycle est une série temporelle. Il faut agréger les signaux en features statistiques par cycle.

### Features à extraire pour PS2 (6000 pts) et FS1 (600 pts) :
- `mean`, `std`, `min`, `max`, `median`
- `q25`, `q75` (quartiles)
- `range` (max - min)
- `rms` (root mean square)
- `skew`, `kurtosis`

### Résultat attendu
Un DataFrame de forme `(n_cycles, n_features)` où chaque ligne correspond à un cycle, et les colonnes sont les features extraites de PS2 et FS1.

---

## Modélisation

### Approche recommandée
1. Tester plusieurs modèles : `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier` (ou XGBoost)
2. Utiliser la **validation croisée** (`StratifiedKFold`, 5 folds) sur le train set uniquement
3. Optimiser les hyperparamètres avec `GridSearchCV` ou `RandomizedSearchCV`
4. Sélectionner le meilleur modèle selon le **F1-score** (métrique principale)

### Métriques d'évaluation
- Accuracy
- F1-score (macro)
- Precision / Recall
- Matrice de confusion
- ROC-AUC

### Sérialisation
- Sauvegarder le modèle final avec `joblib` dans `models/model_v1.pkl`
- Versionner avec **DVC**

---

## Application Web (FastAPI)

### Endpoint principal
```
GET /predict?cycle_id={int}
```
- Entrée : numéro de cycle (entier)
- Sortie JSON :
```json
{
  "cycle_id": 42,
  "prediction": "optimal",
  "probability": 0.93
}
```

### Endpoints complémentaires
```
GET /health          → status de l'API
GET /model/info      → version du modèle, métriques d'entraînement
```

### Stack recommandée
- **FastAPI** + **Uvicorn**
- Charger le modèle au démarrage (`@app.on_event("startup")`)
- Les données des cycles sont pré-chargées en mémoire (ou lues depuis `data/processed/`)

---

## Tests Unitaires

Utiliser **pytest**. Couvrir a minima :

| Fichier de test         | Ce qu'il doit tester                                              |
|-------------------------|-------------------------------------------------------------------|
| `test_load_data.py`     | Chargement correct des fichiers, shape attendue, pas de NaN       |
| `test_preprocess.py`    | Feature engineering, split train/test respecte les 2000 cycles    |
| `test_predict.py`       | Prédiction sur un cycle connu retourne 0 ou 1                     |
| `test_api.py`           | Endpoint `/predict` retourne 200 avec le bon format JSON          |

---

## Docker

### `Dockerfile`
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `docker-compose.yml`
- Service `api` : l'application FastAPI
- Service `grafana` : monitoring (port 3000)
- Service `prometheus` (optionnel) : collecte des métriques

---

## CI/CD (GitHub Actions)

### Pipeline `.github/workflows/ci_cd.yml`
Déclenché sur chaque `push` et `pull_request` vers `main` :

1. **Lint** : `flake8` ou `ruff`
2. **Tests** : `pytest tests/`
3. **Build Docker** : `docker build`
4. **Push image** : vers Docker Hub ou GitHub Container Registry (sur `main` uniquement)

---

## Versioning des données et du modèle (DVC)

- Initialiser DVC : `dvc init`
- Tracker les données brutes : `dvc add data/raw/`
- Tracker le modèle : `dvc add models/`
- Remote storage : S3, GCS, ou DVC remote local pour les tests
- `dvc.yaml` définit le pipeline : load → preprocess → train → evaluate

---

## Monitoring (Grafana)

### Métriques à monitorer
- Nombre de prédictions par heure
- Distribution des prédictions (ratio optimal / non optimal)
- Latence des requêtes API
- Drift potentiel (si nouvelles données disponibles)

### Stack
- **Prometheus** collecte les métriques exposées par FastAPI (`/metrics` via `prometheus-fastapi-instrumentator`)
- **Grafana** visualise les dashboards

---

## Conventions de code

- **Python 3.11+**
- **Style** : PEP8, formaté avec `black`, linting avec `ruff`
- **Typage** : utiliser les type hints Python partout
- **Logging** : utiliser le module `logging` standard (pas de `print`)
- **Docstrings** : Google style pour toutes les fonctions publiques
- **Pas de données brutes dans Git** : utiliser `.gitignore` + DVC

---

## Ordre d'exécution recommandé

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Charger et préparer les données
python src/data/load_data.py
python src/data/preprocess.py

# 3. Entraîner le modèle
python src/models/train.py

# 4. Évaluer sur le test set final
python src/models/evaluate.py

# 5. Lancer l'API
uvicorn api.app:app --reload

# 6. Lancer les tests
pytest tests/

# 7. Lancer via Docker
docker-compose up --build
```

---

## Rappels importants pour Claude Code

- ⚠️ Le split train/test est **fixe** : 2000 premiers cycles = train, le reste = test. Ne jamais le modifier.
- ⚠️ La variable cible est dans `profile.txt`, colonne d'index **3** (0-indexé).
- ⚠️ Les fichiers PS2 et FS1 n'ont **pas d'en-tête**.
- ⚠️ La métrique principale est le **F1-score**, pas l'accuracy (classes potentiellement déséquilibrées).
- ⚠️ Toujours sauvegarder le modèle après entraînement dans `models/`.
- ✅ Justifier tous les choix méthodologiques dans le notebook ou en commentaires.
