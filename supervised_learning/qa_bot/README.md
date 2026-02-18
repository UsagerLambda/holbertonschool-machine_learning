# QA Bot

## Description

Ce projet implémente un bot de questions-réponses (QA Bot) en utilisant BERT et le Universal Sentence Encoder. Le bot est capable de répondre à des questions en se basant sur un corpus de documents Markdown, en combinant la recherche sémantique et l'extraction de réponses.

## Technologies

- Python 3.10
- TensorFlow / TensorFlow Hub
- Transformers (HuggingFace) - modèle `bert-large-uncased-whole-word-masking-finetuned-squad`
- Universal Sentence Encoder (Google)
- NumPy

## Fichiers

| Fichier | Description |
|---|---|
| `0-qa.py` | Fonction `question_answer(question, reference)` qui extrait une réponse à partir d'une question et d'un texte de référence en utilisant BERT |
| `1-loop.py` | Boucle interactive de questions-réponses sans réponse (script de base) |
| `2-qa.py` | Fonction `answer_loop(reference)` - boucle interactive qui répond aux questions en utilisant un texte de référence unique |
| `3-semantic_search.py` | Fonction `semantic_search(corpus_path, sentence)` qui trouve le document le plus pertinent dans un corpus via le Universal Sentence Encoder |
| `4-qa.py` | Fonction `question_answer(corpus_path)` - boucle interactive combinant la recherche sémantique et l'extraction de réponses sur un corpus complet |

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Initialisation en une seule commande :

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

## Utilisation

```bash
./0-main.py   # Test basique de question-réponse
./1-loop.py   # Boucle sans réponse
./2-main.py   # Boucle QA sur un seul document
./3-main.py   # Test de recherche sémantique
./4-main.py   # Bot complet avec recherche sémantique + QA
```

Pour quitter la boucle interactive, tapez : `exit`, `quit`, `goodbye` ou `bye`.
