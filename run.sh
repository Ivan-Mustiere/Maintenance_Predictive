#!/usr/bin/env bash
set -euo pipefail

PYTHON=$(command -v python3 || command -v python)
export PYTHONPATH="$(pwd)"

usage() {
    echo "Usage: ./run.sh [commande]"
    echo ""
    echo "Commandes disponibles :"
    echo "  install      Installer les dependances"
    echo "  preprocess   Feature engineering + split train/test"
    echo "  train        Entrainer le modele"
    echo "  evaluate     Evaluer sur le test set"
    echo "  pipeline     preprocess + train + evaluate"
    echo "  api          Lancer l'API FastAPI (port 8000)"
    echo "  mlflow       Lancer l'interface MLflow (port 5000)"
    echo "  test         Lancer les tests pytest"
    echo "  docker       Lancer via docker-compose"
    echo "  stop         Arreter les services (API, MLflow, Docker)"
    echo "  all          install + pipeline + test"
    echo ""
}

cmd_install() {
    echo ">>> Installation des dependances..."
    pip install -r requirements.txt
}

cmd_preprocess() {
    echo ">>> Feature engineering..."
    $PYTHON src/data/preprocess.py
}

cmd_train() {
    echo ">>> Entrainement du modele..."
    $PYTHON src/models/train.py
}

cmd_evaluate() {
    echo ">>> Evaluation sur le test set..."
    $PYTHON src/models/evaluate.py
}

cmd_pipeline() {
    cmd_preprocess
    cmd_train
    cmd_evaluate
}

cmd_api() {
    echo ">>> Lancement de l'API sur http://localhost:8000 ..."
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
}

cmd_mlflow() {
    echo ">>> Lancement MLflow UI sur http://localhost:5000 ..."
    mlflow ui --host 0.0.0.0 --port 5000
}

cmd_test() {
    echo ">>> Tests pytest..."
    $PYTHON -m pytest tests/ -v
}

cmd_docker() {
    echo ">>> Lancement via Docker Compose..."
    docker compose up --build
}

cmd_stop() {
    echo ">>> Arret des services..."
    pkill -f "uvicorn api.app:app" 2>/dev/null && echo "API FastAPI arretee." || echo "API FastAPI non trouvee."
    pkill -f "mlflow ui" 2>/dev/null && echo "MLflow arrete." || echo "MLflow non trouve."
    if [ -f docker-compose.yml ]; then
        docker compose down && echo "Docker Compose arrete."
    fi
}

cmd_all() {
    cmd_install
    cmd_pipeline
    cmd_test
}

case "${1:-}" in
    install)    cmd_install ;;
    preprocess) cmd_preprocess ;;
    train)      cmd_train ;;
    evaluate)   cmd_evaluate ;;
    pipeline)   cmd_pipeline ;;
    api)        cmd_api ;;
    mlflow)     cmd_mlflow ;;
    test)       cmd_test ;;
    docker)     cmd_docker ;;
    stop)       cmd_stop ;;
    all)        cmd_all ;;
    *)          usage ;;
esac
