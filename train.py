import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Charger les intentions depuis le fichier JSON
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# Boucle à travers chaque phrase dans les modèles d'intention
for intent in intents['intents']:
    tag = intent['tag']
    # Ajouter à la liste des tags
    tags.append(tag)
    for pattern in intent['patterns']:
        # Tokeniser chaque mot dans la phrase
        w = tokenize(pattern)
        # Ajouter à la liste des mots
        all_words.extend(w)
        # Ajouter aux paires xy
        xy.append((w, tag))

# Stemming et mise en minuscule de chaque mot
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Supprimer les doublons et trier
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Créer les données d'entraînement
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X : sac de mots pour chaque phrase modèle
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y : PyTorch CrossEntropyLoss n'a besoin que des étiquettes de classe, pas de one-hot
    label = tags.index(tag)
    y_train.append(label)

# Conversion des listes en tenseurs PyTorch directement
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# Hyper-paramètres
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

# Création du dataset personnalisé


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # Supporte l'indexation pour que dataset[i] retourne le i-ème échantillon
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # Permet d'utiliser len(dataset) pour obtenir la taille
    def __len__(self):
        return self.n_samples


# Initialiser le DataLoader
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# Choisir l'appareil (GPU si disponible, sinon CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialiser le modèle
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Fonction de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Entraîner le modèle
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # Propagation avant
        outputs = model(words)
        # Calcul de la perte
        loss = criterion(outputs, labels)

        # Rétropropagation et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Affichage de la perte finale
print(f'Final loss: {loss.item():.4f}')

# Sauvegarde des états du modèle
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "model001.pth"
torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')
