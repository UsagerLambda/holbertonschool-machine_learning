import matplotlib.pyplot as plt
from config import LEARNING_RATE, BATCH_SIZE, EPOCHS, DROPOUT_RATE, DENSE_UNITS, MODEL_NAME, PATIENCE, TRAINABLE, NB_UNFREEZE_LAYERS

def plot_training_history(history, save_path='img/training_plots.png'):
    """Crée des graphiques pour visualiser l'entraînement."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Résultats d\'entraînement - {MODEL_NAME}', fontsize=16)

    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_title('Précision du modèle')
    axes[0, 0].set_xlabel('Époque')
    axes[0, 0].set_ylabel('Précision')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Loss
    axes[0, 1].plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 1].set_title('Perte du modèle')
    axes[0, 1].set_xlabel('Époque')
    axes[0, 1].set_ylabel('Perte')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Accuracy en pourcentage
    axes[1, 0].plot([acc*100 for acc in history.history['accuracy']], 'b-', label='Training %', linewidth=2)
    axes[1, 0].plot([acc*100 for acc in history.history['val_accuracy']], 'r-', label='Validation %', linewidth=2)
    axes[1, 0].set_title('Précision en pourcentage')
    axes[1, 0].set_xlabel('Époque')
    axes[1, 0].set_ylabel('Précision (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Différence entre train et val (overfitting detection)
    diff_acc = [t-v for t, v in zip(history.history['accuracy'], history.history['val_accuracy'])]
    axes[1, 1].plot(diff_acc, 'g-', label='Train - Val Accuracy', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Détection d\'overfitting')
    axes[1, 1].set_xlabel('Époque')
    axes[1, 1].set_ylabel('Différence précision')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Graphiques sauvegardés dans : {save_path}")

def print_training_summary(history, training_time, total_time):
    """Affiche un résumé détaillé de l'entraînement."""
    final_train_acc = history.history['accuracy'][-1] * 100
    final_val_acc = history.history['val_accuracy'][-1] * 100
    best_val_acc = max(history.history['val_accuracy']) * 100
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    print("\n" + "="*60)
    print("📊 RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("="*60)
    print(f"🔧 Configuration:")
    print(f"   • Modèle: {MODEL_NAME}")
    print(f"   • Learning Rate: {LEARNING_RATE}")
    print(f"   • Batch Size: {BATCH_SIZE}")
    print(f"   • Epochs: {EPOCHS}")
    print(f"   • Dropout: {DROPOUT_RATE}")
    print(f"   • Dense Units: {DENSE_UNITS}")
    if TRAINABLE == True:
        print(f"   • Unfreezed layers: {NB_UNFREEZE_LAYERS}")
    print(f"   • Patience: {PATIENCE}")

    print(f"\n📈 Résultats finaux:")
    print(f"   • Training Accuracy: {final_train_acc:.2f}%")
    print(f"   • Validation Accuracy: {final_val_acc:.2f}%")
    print(f"   • Meilleure val accuracy: {best_val_acc:.2f}%")
    print(f"   • Training Loss: {final_train_loss:.4f}")
    print(f"   • Validation Loss: {final_val_loss:.4f}")

    overfitting = final_train_acc - final_val_acc
    print(f"\n🔍 Diagnostic:")
    if overfitting > 5:
        print(f"Surapprentissage détecté ! ({overfitting})")
    elif overfitting < -5:
        print(f"Sous-apprentissage détecté ! ({overfitting})")
    else:
        print(f"Bon équilibre ! ({overfitting})")

    print(f"\n⏱️  Temps d'exécution:")
    print(f"   • Temps total: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"   • Temps d'entraînement: {training_time:.2f}s ({training_time/60:.2f}min)")
    print(f"   • Temps par époque: {training_time/len(history.history['loss']):.2f}s")
    print("="*60)
