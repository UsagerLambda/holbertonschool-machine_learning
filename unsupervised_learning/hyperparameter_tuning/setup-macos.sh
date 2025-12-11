#!/bin/bash

# Script de setup pour l'environnement hyperparameter_tuning sur macOS
# Usage: ./setup-macos.sh

set -e  # Arrêter en cas d'erreur

echo "🍎 Configuration de l'environnement hyperparameter_tuning pour macOS..."

# Nom de l'environnement virtuel
VENV_NAME="venv"

# Vérifier si on est bien sur macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Attention: Ce script est optimisé pour macOS"
    echo "   Pour Linux avec GPU NVIDIA, utilisez setup.sh"
    read -p "Continuer quand même? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Vérifier si Python 3 est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Erreur: Python 3 n'est pas installé"
    echo "   Installez-le avec: brew install python@3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✓ Python version: $PYTHON_VERSION"

# Détection de l'architecture macOS
ARCH=$(uname -m)
echo "✓ macOS détecté - Architecture: $ARCH"
if [[ "$ARCH" == "arm64" ]]; then
    echo "  → Apple Silicon (M1/M2/M3) - tensorflow-metal sera installé pour l'accélération GPU"
else
    echo "  → Intel Mac"
fi

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

# Installer les dépendances depuis requirements-macos.txt
echo "📥 Installation des dépendances depuis requirements-macos.txt..."
if [ -f "requirements-macos.txt" ]; then
    pip install -r requirements-macos.txt
else
    echo "❌ Erreur: requirements-macos.txt introuvable"
    exit 1
fi

echo ""
echo "✅ Installation terminée avec succès!"
echo ""
echo "📋 Résumé:"
echo "  - Plateforme: macOS ($ARCH)"
echo "  - Python: $PYTHON_VERSION"
echo "  - Environnement: $VENV_NAME"
echo ""
echo "Pour activer l'environnement virtuel:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "Pour désactiver l'environnement virtuel:"
echo "  deactivate"
echo ""
echo "Pour exécuter votre script:"
echo "  python 6-bayes_opt.py"
echo ""
