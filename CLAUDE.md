# CLAUDE.md – Projet Maintenance Prédictive (Valve Condition)

## Vue d'ensemble du projet

Tu es Data Scientist / ML Engineer dans une entreprise industrielle.
L'objectif est de construire un système de **maintenance prédictive** pour prédire si la condition de la valve d'un cycle de production hydraulique est **optimale (100%)** ou **non optimale**.

- **Type de problème** : Classification binaire supervisée
- **Dataset source** : UCI – Condition Monitoring of Hydraulic Systems
  → https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems
- **2205 cycles** de production, 22 features statistiques extraites de 2 capteurs

---

## Structure du projet

```
projet-maintenance-predictive/
├── CLAUDE.md
├── README.md
├── .gitignore
├── requirements.txt
├── run.sh                        # Script de commandes principal
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml                      # Pipeline DVC
├── .github/
│   └── workflows/
│       └── ci_cd.yml
├── data/
│   ├── raw/                      # Données brutes (non versionnées Git)
│   └── processed/                # Features engineered, prêtes au ML
├── notebooks/
│   └── exploration.ipynb         # EDA + justification des choix
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py          # Chargement des fichiers bruts
│   │   └── preprocess.py         # Feature engineering + split train/test
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py              # Entraînement + tracking MLflow
│   │   ├── evaluate.py           # Évaluation sur le test set final
│   │   └── predict.py            # Prédiction à partir d'un numéro de cycle
│   └── utils/
│       ├── __init__.py
│       └── logger.py
├── api/
│   ├── __init__.py
│   └── app.py                    # Application web FastAPI
├── tests/
│   ├── __init__.py
│   ├── test_load_data.py
│   ├── test_preprocess.py
│   ├── test_predict.py
│   └── test_api.py
├── models/                       # Modèles sérialisés (.pkl), versionnés via DVC
│   ├── model_v1.pkl
│   ├── train_metrics.json
│   └── eval_metrics.json
└── monitoring/
    ├── prometheus.yml
    └── grafana/
        └── dashboard.json        # Dashboard Grafana prêt à l'emploi
```

---

## Données

### Fichiers fournis (dans `data/raw/`)

| Fichier | Description | Fréquence |
|---|---|---|
| `PS2.txt` | Pression (bar) | 100 Hz → 6000 valeurs/cycle |
| `FS1.txt` | Débit volumique (L/min) | 10 Hz → 600 valeurs/cycle |
| `profile.txt` | Variables cibles (5 colonnes) | 1 valeur/cycle |

### Format des fichiers
- Séparateur : tabulation (`\t`)
- Pas d'en-tête
- Chaque ligne = 1 cycle de production

### Colonnes de `profile.txt` (0-indexé)

| Index | Variable | Valeurs |
|---|---|---|
| 0 | Cooler condition | 3, 20, 100 |
| **1** | **Valve condition** ← **cible** | **100, 90, 80, 73** |
| 2 | Internal pump leakage | 0, 1, 2 |
| 3 | Hydraulic accumulator | 130, 115, 100, 90 |
| 4 | Stable flag | 0, 1 |

### Variable cible : `valve condition` (colonne index **1**)
- `100` → condition optimale → **classe 1** (1125 cycles)
- `90 / 80 / 73` → non optimale → **classe 0** (1080 cycles)

> ⚠️ **Attention** : la colonne index 3 est le hydraulic accumulator, PAS la valve condition.

### Split train/test (contrainte stricte)
- **Train** : les **2000 premiers cycles** (index 0 à 1999)
- **Test final** : les **cycles restants** (index 2000 à 2204)
- Ne jamais utiliser le test set pendant l'entraînement ou la validation croisée

---

## Feature Engineering

Chaque cycle est une série temporelle agrégée en **22 features statistiques** (11 par capteur).

### Features extraites pour PS2 et FS1 :
`mean`, `std`, `min`, `max`, `median`, `q25`, `q75`, `range`, `rms`, `skew`, `kurtosis`

### Résultat
DataFrame de forme `(2205, 22)` — chaque ligne = 1 cycle.

---

## Modélisation

### Approche
1. 3 modèles candidats : `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`
2. `StratifiedKFold(5)` sur le train set uniquement
3. `RandomizedSearchCV` pour l'optimisation des hyperparamètres
4. Sélection par **F1-macro** (métrique principale)
5. Tracking de chaque run avec **MLflow**

### Résultats obtenus

| Modèle | CV F1-macro |
|---|---|
| Random Forest | **0.9975** ✓ sélectionné |
| Gradient Boosting | 0.9970 |
| Logistic Regression | 0.9965 |

**Test set** : F1-macro=1.0, Accuracy=1.0, ROC-AUC=1.0
(cohérent avec la doc UCI : *"Valve state is an easy target"*)

### Sérialisation
- `models/model_v1.pkl` — modèle joblib
- `models/train_metrics.json` — métriques CV + run MLflow ID
- `models/eval_metrics.json` — métriques test set

---

## Application Web (FastAPI)

### Endpoints

| Endpoint | Description |
|---|---|
| `GET /predict?cycle_id={int}` | Prédit la condition de la valve |
| `GET /health` | Statut de l'API + uptime |
| `GET /model/info` | Version, métriques train + test |
| `GET /metrics` | Métriques Prometheus |

### Réponse `/predict`
```json
{
  "cycle_id": 42,
  "prediction": "optimal",
  "probability": 0.97
}
```

### Stack
- **FastAPI** + **Uvicorn**
- Modèle et features pré-chargés au démarrage via `lifespan`
- Métriques exposées via `prometheus-fastapi-instrumentator`

---

## MLflow

Chaque entraînement logue automatiquement dans l'expérience `valve-condition` :
- **3 runs** de training (un par modèle candidat) : hyperparamètres + CV F1-macro + modèle artifact
- **1 run** `evaluation` : métriques test set complètes

```bash
./run.sh mlflow   # → http://localhost:5000
```

---

## Tests Unitaires (29 tests)

| Fichier | Ce qu'il teste |
|---|---|
| `test_load_data.py` | Shape, absence de NaN, cohérence des cycles |
| `test_preprocess.py` | Features keys/values, mapping cible, split 2000/reste |
| `test_predict.py` | Format de sortie, label binaire, probabilité ∈ [0,1] |
| `test_api.py` | Codes HTTP 200/404/422, format JSON des réponses |

```bash
./run.sh test
```

---

## Docker

```bash
./run.sh docker
```

| Service | Port |
|---|---|
| API FastAPI | 8000 |
| Prometheus | 9090 |
| Grafana | 3000 (admin/admin) |

---

## CI/CD (GitHub Actions)

Déclenché sur chaque `push` et `pull_request` vers `main` :

1. **Lint** : `ruff` + `black --check`
2. **Tests** : `pytest tests/`
3. **Build Docker** : `docker build`
4. **Push image** : Docker Hub (sur `main` uniquement)

Secrets requis : `DOCKER_USERNAME`, `DOCKER_PASSWORD`

---

## Ordre d'exécution

```bash
./run.sh install     # Installer les dépendances
./run.sh pipeline    # preprocess + train + evaluate
./run.sh api         # Lancer l'API → http://localhost:8000
./run.sh mlflow      # Voir les expériences → http://localhost:5000
./run.sh test        # Lancer les tests
./run.sh docker      # Tout via Docker Compose
```

---

## Conventions de code

- **Python 3.11+**
- **Style** : PEP8, formaté avec `black`, linting avec `ruff`
- **Typage** : type hints partout
- **Logging** : module `logging` standard (pas de `print`)
- **Docstrings** : Google style pour toutes les fonctions publiques
- **Pas de données brutes dans Git** : `.gitignore` + DVC

---

## Rappels importants pour Claude Code

- ⚠️ Le split train/test est **fixe** : 2000 premiers cycles = train, le reste = test. Ne jamais le modifier.
- ⚠️ La variable cible est dans `profile.txt`, colonne d'index **1** (0-indexé) — valve condition.
- ⚠️ La colonne index **3** est le hydraulic accumulator — ne pas confondre.
- ⚠️ Les fichiers PS2 et FS1 n'ont **pas d'en-tête**.
- ⚠️ La métrique principale est le **F1-macro**.
- ⚠️ `mlruns/` et `mlartifacts/` sont dans `.gitignore` — ne pas versionner.
- ✅ Chaque entraînement logue automatiquement dans MLflow (expérience `valve-condition`).
- ✅ Justifier tous les choix méthodologiques dans le notebook ou en commentaires.
