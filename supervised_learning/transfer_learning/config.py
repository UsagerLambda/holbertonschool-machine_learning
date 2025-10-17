# config.py
# Centralisation des hyperparamètres et variables de configuration

LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 5
DROPOUT_RATE = 0.3
DENSE_UNITS = 256
MODEL_NAME = 'EfficientNetV2B1'  # 'MobileNetV2' 'ResNet50'
PATIENCE = 5
TRAINABLE = True
NB_UNFREEZE_LAYERS = 15  # Nombre de couches à dégeler (TRAINABLE doit être True)
PLOT = True
