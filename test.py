import torch

# Vérifier si CUDA est disponible
print(torch.cuda.is_available())

# Afficher le nombre de GPU disponibles
print(torch.cuda.device_count())

# Afficher le nom du GPU
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))