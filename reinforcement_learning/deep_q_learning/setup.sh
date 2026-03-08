#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="3.9.19"
VENV_NAME="breakout"

# --- Vérifier si pyenv est installé ---
if ! command -v pyenv &>/dev/null; then
    echo "pyenv n'est pas installé."
    read -p "Installer pyenv ? (o/n) " choice
    [[ "$choice" != "o" ]] && echo "Abandon." && exit 1

    curl -fsSL https://pyenv.run | bash

    # Ajouter pyenv au shell courant
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"

    # Persister dans .bashrc si pas déjà fait
    if ! grep -q 'pyenv init' ~/.bashrc 2>/dev/null; then
        {
            echo ''
            echo '# pyenv'
            echo 'export PYENV_ROOT="$HOME/.pyenv"'
            echo 'export PATH="$PYENV_ROOT/bin:$PATH"'
            echo 'eval "$(pyenv init -)"'
        } >> ~/.bashrc
    fi

    echo "pyenv installé."
fi

eval "$(pyenv init -)"

# --- Vérifier si pyenv-virtualenv est installé ---
if ! pyenv commands | grep -q virtualenv; then
    echo "pyenv-virtualenv n'est pas installé."
    read -p "Installer pyenv-virtualenv ? (o/n) " choice
    [[ "$choice" != "o" ]] && echo "Abandon." && exit 1

    git clone https://github.com/pyenv/pyenv-virtualenv.git "$(pyenv root)/plugins/pyenv-virtualenv"
    eval "$(pyenv virtualenv-init -)"

    if ! grep -q 'virtualenv-init' ~/.bashrc 2>/dev/null; then
        echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
    fi

    echo "pyenv-virtualenv installé."
fi

eval "$(pyenv virtualenv-init -)" 2>/dev/null || true

# --- Installer Python si nécessaire ---
if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
    echo "Installation de Python ${PYTHON_VERSION}..."
    pyenv install "$PYTHON_VERSION"
else
    echo "Python ${PYTHON_VERSION} déjà disponible."
fi

# --- Créer le virtualenv ---
if pyenv versions --bare | grep -q "^${VENV_NAME}$"; then
    echo "Le virtualenv '${VENV_NAME}' existe déjà."
else
    echo "Création du virtualenv '${VENV_NAME}'..."
    pyenv virtualenv "$PYTHON_VERSION" "$VENV_NAME"
fi

# --- Activer et installer les dépendances ---
pyenv activate "$VENV_NAME"

if [[ -f requirements.txt ]]; then
    echo "Installation des dépendances..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "Done. Dépendances installées."
else
    echo "Pas de requirements.txt trouvé dans le répertoire courant."
fi

echo ""
echo "Pour activer l'env : pyenv activate ${VENV_NAME}"
echo "Pour le set local  : pyenv local ${VENV_NAME}"
