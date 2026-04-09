# Maintenance Prédictive – Condition de Valve Hydraulique

Classification binaire pour prédire si la condition de valve d'un système hydraulique est **optimale (100%)** ou **non optimale**.

**Dataset** : [UCI – Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)

---

## Démarrage rapide

```bash
# 1. Installer les dépendances
./run.sh install

# 2. Placer les données brutes dans data/raw/
#    (PS2.txt, FS1.txt, profile.txt)

# 3. Pipeline complet (preprocess + train + evaluate)
./run.sh pipeline

# 4. Lancer l'API
./run.sh api

# 5. Tests
./run.sh test
```

### Commandes disponibles

| Commande | Description |
|---|---|
| `./run.sh install` | Installer les dépendances |
| `./run.sh preprocess` | Feature engineering + split train/test |
| `./run.sh train` | Entraîner le modèle |
| `./run.sh evaluate` | Evaluer sur le test set |
| `./run.sh pipeline` | preprocess + train + evaluate |
| `./run.sh api` | Lancer l'API FastAPI (port 8000) |
| `./run.sh mlflow` | Lancer l'interface MLflow (port 5000) |
| `./run.sh test` | Lancer les tests pytest |
| `./run.sh docker` | Lancer via docker-compose |
| `./run.sh all` | install + pipeline + test |

---

## Via Docker

```bash
./run.sh docker
```

| Service | URL |
|---|---|
| API FastAPI | http://localhost:8000 |
| MLflow UI | http://localhost:5000 |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |

---

## API

| Endpoint | Description | Exemple de réponse |
|---|---|---|
| `GET /predict?cycle_id=42` | Prédit la condition de la valve | `{"cycle_id": 42, "prediction": "optimal", "probability": 0.97}` |
| `GET /health` | Statut de l'API | `{"status": "ok", "uptime_seconds": 42.0}` |
| `GET /model/info` | Version et métriques du modèle | `{"version": "v1", "best_model": "random_forest", ...}` |
| `GET /metrics` | Métriques Prometheus | — |

---

## Structure du projet

```
├── data/
│   ├── raw/           # PS2.txt, FS1.txt, profile.txt (non versionnés Git)
│   └── processed/     # Features engineered, prêtes au ML
├── notebooks/
│   └── exploration.ipynb   # EDA + justification des choix
├── src/
│   ├── data/
│   │   ├── load_data.py    # Chargement des fichiers bruts
│   │   └── preprocess.py   # Feature engineering + split train/test
│   ├── models/
│   │   ├── train.py        # Entraînement + tracking MLflow
│   │   ├── evaluate.py     # Evaluation sur le test set final
│   │   └── predict.py      # Prédiction par cycle_id
│   └── utils/
│       └── logger.py
├── api/
│   └── app.py              # Application FastAPI
├── tests/                  # Tests unitaires pytest (29 tests)
├── models/                 # Modèle sérialisé (model_v1.pkl)
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboard.json  # Dashboard Grafana prêt à l'emploi
├── .github/workflows/
│   └── ci_cd.yml           # CI/CD GitHub Actions
├── dvc.yaml                # Pipeline DVC
├── Dockerfile
├── docker-compose.yml
└── run.sh                  # Script de commandes
```

---

## Feature Engineering

Chaque cycle est une série temporelle agrégée en **22 features statistiques** (11 par capteur) :

| Capteur | Fréquence | Points/cycle | Features |
|---|---|---|---|
| PS2 (pression) | 100 Hz | 6000 | mean, std, min, max, median, q25, q75, range, rms, skew, kurtosis |
| FS1 (débit) | 10 Hz | 600 | mean, std, min, max, median, q25, q75, range, rms, skew, kurtosis |

**Variable cible** : `profile.txt` colonne index 1 — valve condition  
→ `100` = optimal → classe 1 | `90 / 80 / 73` = non optimal → classe 0

**Split train/test fixe** : 2000 premiers cycles = train, cycles 2000+ = test

---

## Modélisation

3 modèles candidats évalués en **StratifiedKFold(5)**, sélection par **F1-macro** :

| Modèle | CV F1-macro |
|---|---|
| Random Forest | **0.9975** ✓ |
| Gradient Boosting | 0.9970 |
| Logistic Regression | 0.9965 |

**Résultats test set (205 cycles) :**

| Métrique | Score |
|---|---|
| F1-macro | 1.00 |
| Accuracy | 1.00 |
| ROC-AUC | 1.00 |

> Les scores parfaits sont cohérents avec la documentation UCI : *"Valve state is an easy target (perfect classification achieved)"*.

---

## MLflow

Les expériences sont trackées automatiquement à chaque `./run.sh train` et `./run.sh evaluate`.

```bash
./run.sh mlflow   # → http://localhost:5000
```

Chaque run enregistre les hyperparamètres, le CV F1-macro et le modèle sérialisé.  
Le run `evaluation` enregistre les métriques finales sur le test set.

---

## CI/CD

Pipeline GitHub Actions déclenché sur chaque push/PR vers `main` :

1. **Lint** — `ruff` + `black`
2. **Tests** — `pytest tests/`
3. **Build Docker** — `docker build`
4. **Push image** — vers Docker Hub (sur `main` uniquement)

Secrets requis : `DOCKER_USERNAME`, `DOCKER_PASSWORD`
