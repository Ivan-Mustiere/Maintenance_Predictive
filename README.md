# Maintenance Prédictive – Condition de Valve Hydraulique

Classification binaire pour prédire si la condition de valve d'un système hydraulique est **optimale (100%)** ou **non optimale**.

**Dataset** : [UCI – Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)

---

## Démarrage rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Placer les données brutes dans data/raw/
#    (PS2.txt, FS1.txt, profile.txt)

# 3. Pipeline complet
python src/data/preprocess.py
python src/models/train.py
python src/models/evaluate.py

# 4. Lancer l'API
uvicorn api.app:app --reload

# 5. Tests
pytest tests/ -v
```

## Via Docker

```bash
docker-compose up --build
```

- API : http://localhost:8000
- Grafana : http://localhost:3000 (admin/admin)
- Prometheus : http://localhost:9090

## API

| Endpoint | Description |
|----------|-------------|
| `GET /predict?cycle_id=42` | Prédit la condition de la valve |
| `GET /health` | Statut de l'API |
| `GET /model/info` | Métriques du modèle |
| `GET /metrics` | Métriques Prometheus |

## Structure

```
src/data/      → chargement + feature engineering
src/models/    → entraînement, évaluation, prédiction
api/           → application FastAPI
tests/         → tests unitaires pytest
monitoring/    → configs Prometheus + Grafana
```

## Métriques cibles

| Métrique | Description |
|----------|-------------|
| F1-macro | Métrique principale |
| ROC-AUC | Discrimination globale |
| Accuracy | Référence |
