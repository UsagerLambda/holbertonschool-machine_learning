#!/bin/bash

# Script de setup pour l'environnement hyperparameter_tuning
# Usage: ./setup.sh

set -e  # Arrêter en cas d'erreur

echo "🔧 Configuration de l'environnement hyperparameter_tuning..."

# Nom de l'environnement virtuel
VENV_NAME="venv"

# Vérifier si Python 3 est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Erreur: Python 3 n'est pas installé"
    exit 1
fi

echo "✓ Python version: $(python3 --version)"

# Créer l'environnement virtuel s'il n'existe pas
if [ ! -d "$VENV_NAME" ]; then
    echo "📦 Création de l'environnement virtuel '$VENV_NAME'..."
    python3 -m venv "$VENV_NAME"
else
    echo "✓ L'environnement virtuel '$VENV_NAME' existe déjà"
fi

# Activer l'environnement virtuel
echo "🔌 Activation de l'environnement virtuel..."
source "$VENV_NAME/bin/activate"

# Mettre à jour pip
echo "⬆️  Mise à jour de pip..."
pip install --upgrade pip

# Installer les dépendances
echo "📥 Installation des dépendances depuis requirements.txt..."
pip install -r requirements.txt

echo ""
echo "✅ Installation terminée avec succès!"
echo ""
echo "Pour activer l'environnement virtuel:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "Pour désactiver l'environnement virtuel:"
echo "  deactivate"
echo ""
echo "Pour exécuter votre script:"
echo "  python 6-bayes_opt.py"
