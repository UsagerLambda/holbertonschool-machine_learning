# converter.py
import tensorflow as tf
from tensorflow import keras as K

# Wrappers pour charger
class GlorotUniformCompat(K.initializers.GlorotUniform):
    def __init__(self, seed=None, **kwargs):
        kwargs.pop('dtype', None)
        super().__init__(seed=seed)

class ZerosCompat(K.initializers.Zeros):
    def __init__(self, **kwargs):
        kwargs.pop('dtype', None)
        super().__init__()

class OnesCompat(K.initializers.Ones):
    def __init__(self, **kwargs):
        kwargs.pop('dtype', None)
        super().__init__()

custom_objects = {
    'GlorotUniform': GlorotUniformCompat,
    'Zeros': ZerosCompat,
    'Ones': OnesCompat,
}

print("Chargement du modèle...")
model = K.models.load_model('data/yolo.h5', custom_objects=custom_objects, compile=False)
print("✓ Modèle chargé!")

# Sauvegarder les poids
weights = model.get_weights()

# Obtenir la config
config = model.get_config()

def fix_initializer_config(obj):
    """Remplace les initializers custom par les vrais"""
    if isinstance(obj, dict):
        # Si c'est un initializer, remplacer le nom de classe
        if 'class_name' in obj and 'config' in obj:
            class_name = obj['class_name']
            
            # Mapper les noms custom aux vrais noms
            name_mapping = {
                'GlorotUniformCompat': 'GlorotUniform',
                'ZerosCompat': 'Zeros',
                'OnesCompat': 'Ones'
            }
            
            if class_name in name_mapping:
                obj['class_name'] = name_mapping[class_name]
                obj['module'] = 'keras.initializers'
                obj['registered_name'] = None
            
            # Retirer dtype si présent
            if 'dtype' in obj['config']:
                del obj['config']['dtype']
        
        # Récursion sur tous les champs
        for key, value in obj.items():
            obj[key] = fix_initializer_config(value)
    
    elif isinstance(obj, list):
        return [fix_initializer_config(item) for item in obj]
    
    return obj

print("Nettoyage de la configuration...")
config = fix_initializer_config(config)

# Reconstruire le modèle
print("Création du nouveau modèle...")
new_model = K.Model.from_config(config)

# Restaurer les poids
print("Restauration des poids...")
new_model.set_weights(weights)

print("Sauvegarde...")
new_model.save('data/yolo_new.h5')
print("✓ Conversion réussie!")
print("\nMaintenant exécute:")
print("  mv data/yolo.h5 data/yolo_old.h5")
print("  mv data/yolo_new.h5 data/yolo.h5")
print("  ./0-main.py")
