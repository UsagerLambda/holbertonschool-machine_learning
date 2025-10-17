#!/bin/bash

# Vérifier Python dès le début
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python non trouvé"
    exit 1
fi

# Créer dossier img si nécessaire
[ ! -d "img" ] && mkdir img

# Demander quel script lancer
echo ""
echo "Quel script voulez-vous lancer ?"
echo "1) 0-transfer.py (Entraînement)"
echo "2) 0-main.py (Évaluation)"
echo "3) Installer les dépendances"
read -p "Choix (1,2 ou 3): " choice

case $choice in
    1)
        echo "🚀 Lancement de l'entraînement..."
        $PYTHON_CMD 0-transfer.py
        ;;
    2)
        if [ ! -f "cifar10.h5" ]; then
            echo "❌ Modèle cifar10.h5 non trouvé. Lancez d'abord l'entraînement."
            exit 1
        fi
        echo "📊 Lancement de l'évaluation..."
        $PYTHON_CMD 0-main.py
        ;;
    3)
        echo "✅ Python détecté: $($PYTHON_CMD --version)"

        echo "📦 Installation des dépendances..."
        $PYTHON_CMD -m pip install -r requirements.txt
        if [[ "$OSTYPE" == "darwin"* ]] && [[ "$(uname -m)" == "arm64" ]]; then
            echo "🍎 macOS Apple Silicon détecté - Installation de tensorflow-metal..."
            $PYTHON_CMD -m pip install -r requirements-macos.txt
        fi
        ;;
    *)
        echo "❌ Choix invalide"
        exit 1
        ;;
esac
