import pandas as pd
import numpy as np
import keras
import tensorflow
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- PARTIE 1 : Chargement des données ---

# Charger les données avec le chemin correct

filename = "/kaggle/input/nlpcsv/train2.csv"

# Charger le fichier en brut
df = pd.read_csv(filename, header=None)

# Remplacer les valeurs NaN par des chaînes vides
df[0] = df[0].fillna("")

# Découper la colonne unique en deux colonnes sur le point-virgule
df = df[0].str.split(";", expand=True)

# Renommer les colonnes
df.columns = ["Text", "Emotions"]

# Afficher les premières lignes pour vérifier
print(df.head())
print(df.shape)  # Vérifier la forme du DataFrame

# --- PARTIE 2 : Préparation des données ---
texts = df["Text"].tolist()
labels = df["Emotions"].tolist()

# Tokenisation
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Encodage des labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
one_hot_labels = keras.utils.to_categorical(labels)

# Division des données
xtrain, xtest, ytrain, ytest = train_test_split(padded_sequences, one_hot_labels, test_size=0.2)

# --- PARTIE 3 : Définition et entraînement du modèle ---
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, 
                    output_dim=128, input_length=max_length))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=len(one_hot_labels[0]), activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Entraîner le modèle
history = model.fit(xtrain, ytrain, epochs=10, batch_size=32, validation_data=(xtest, ytest))

# --- PARTIE 4 : Visualisations ---

# 1️⃣ Évolution des performances
plt.figure(figsize=(12, 5))

# Graphique de la perte
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Graphique de la précision
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 2️⃣ Distribution des émotions
sns.countplot(x=df["Emotions"])
plt.title("Distribution of Emotions in the Dataset")
plt.xlabel("Emotions")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# 3️⃣ Nuage de mots
text = " ".join(df["Text"].tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Texts")
plt.show()

# 4️⃣ Matrice de confusion
y_pred = model.predict(xtest)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(ytest, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# 5️⃣ Exemples de prédictions
test_texts = ["I am so thrilled today!", "This is the worst day ever."]
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_padded = pad_sequences(test_sequences, maxlen=max_length)

predictions = model.predict(test_padded)
for text, pred in zip(test_texts, predictions):
    print(f"Text: {text}")
    print(f"Predicted Emotion: {label_encoder.inverse_transform([np.argmax(pred)])}")
