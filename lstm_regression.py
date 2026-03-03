""""
Problem tanımı : yorumlardan puan tahmini(1-5 arasında), regresyon problemi
- cok iyiydi cok memnun kaldım -> 4.5
- berbattı, bir daha gelmem -> 1.2

veri seti: yelp dataset, hugging face, (restoran, doktor, otel, araba yıkama....)

lstm : bir yorumu bastan sona okur sonrasında yorumun genel anlamına karsılık gelen yıldız puanını çıkarır

"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.preprocessing.text import Tokenizer 


from tensorflow.keras.preprocessing.sequence import pad_sequences


from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

# load yelp dataset

import pandas as pd

splits = {'train': 'yelp_review_full/train-00000-of-00001.parquet', 'test': 'yelp_review_full/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/Yelp/yelp_review_full/" + splits["train"])


print(df.head())
# etiketleri 0 4 arasndan 1-5 aralığına çekme
df['label'] = df['label'] + 1

# data preprocessing
texts = df['text'].values # yorum metinleri
labels = df['label'].values # yıldız puanlarımız

# tokenizer: metni sayıya cevir
# num words : en cok gecen 10000 kelime
# OOV : bilinmeyen kelimeleri bu etiketle göster

tokenizer = Tokenizer(num_words = 10000, oov_token = "<OOV>")

# Metni sayılara dönüştür
tokenizer.fit_on_texts(texts)

# tokenizerı diske kaydet
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# yorumları dizi haline getirelim
sequences =  tokenizer.texts_to_sequences(texts)

# tüm dizileri sabit uzunluğa getir padding uygulayalım
padded_sequences = pad_sequences(sequences, maxlen = 100, padding = "post", truncating = "post")

# etiketler 0 ile 5 arasında, normalization ile 0-1 arasında alalım
# regresyon problemlerinde daha stabil bir öğrenme sağlıyor
scaler = MinMaxScaler()
labels_scaled = scaler.fit_transform(labels.reshape(-1, 1))

# eğitim ve test verisine ayıralım
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels_scaled,
                                                    test_size= 0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_train shape: {X_train[:2]}")
print(f"y_train shape: {y_train.shape}")
print(f"y_train shape: {y_train[:2]}")

# lstm tabanlı regresyon modeli
model = Sequential()

# embedding katmanı : kelime indekslereini vektör uzaylarına dönüştürür
# input _dim : kelime sayısı
# output_dim : her bir kelime 128 boyutlu vektörle temsil edilecektir
# input_length : sabit dizi uzunluğu yani her bir metnin uzunluğu
model.add(Embedding(input_dim = 10000, output_dim = 128, input_length = 100))

# lstm katmanı
model.add(LSTM(128)) # 128 lstm de bulunan hücre sayısı yani daha fazla öğrenme kapasitesi

# tam bağlı katman (dense) layer
model.add(Dense(64, activation = "relu"))

# output layer 
model.add(Dense(1, activation = "sigmoid")) # relu , tanh, sigmoid, softmax, linear

# model compile and training
model.compile(
    optimizer = "adam", # adaptif öğrenme fonksiyonu
    loss = MeanSquaredError(), # regresyon için loss fonk
    metrics = [MeanAbsoluteError()] # modelin hata ort
)

history = model.fit(
    X_train, y_train,
    epochs = 3,
    batch_size = 64, # her adımda işlenicek örnek sayisi
    validation_split = 0.2 # eğitim verisinin %20 si validasyon için ayrılır
)

# eğitim kayıp grafiğini görselleştir ve modeli kaydet
plt.plot(history.history["loss"], label = "Training Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.title("Eğitim süreci MSE")
plt.xlabel("Epoch")
plt.ylabel("Loss MSE")
plt.show()

# modeli kaydet
model.save("regression_lstm_yelp.h5")


