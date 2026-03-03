""" 
yorum yazalım, lstm bu yorumu tahmin etsin

"""

import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# modeli yükle
model = load_model("regression_lstm_yelp.h5", compile = False)

# tokenizer yükle
with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

# örnek input "ben bir doktora gittim ve bu doktoru cok sevdim"
# 1 star - 5 star
texts = [
    "We went on a weeknight. Place was not busy waited over 20 minutes for drinks and to have our order taken. We ordered an app and it came out with the meals and that was another 20 minutes or so. Food was luke warm at best. I would not go back.",
    "Dave and busters is the best place to eat and play games. There is many fun games and prizes to win. I highly suggest if you want to plays games, go Wednesday when all the games are half off. Food is pretty good and a nice bar. The downside is that the booze is pretty pricey."
]

# tokenizer " 0 1 2 3 4 5 6 7 8 "
# padding "0 1 2 3 4 5 6 7 8 0 0 0 0 0 0" boyu 100 olucak
# text i sayılara cevir ve padding islemini gerçekleştirir.

sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen = 100, padding = "post", truncating = "post")

# lstm prediction
predictions = model.predict(padded)

# sonucları yazdır
print("-" * 50)
for i, comment in enumerate(texts):
    raw_score = predictions[i][0] # 0 ile 1 arasındaki ham puan
    star_score = (raw_score * 4) + 1 # 1 ile 5 arasına çevrilmiş puan
    
    print(f"Yorum: {comment[:60]}...") # Yorumun başını göster
    print(f"Modelin Ham Çıktısı (0-1): {raw_score:.4f}")
    print(f"Tahmini Yıldız (1-5): {star_score:.1f}") # Dönüştürülmüş gerçek yıldız
    print("-" * 50)
    
    bar = "★" * int(round(star_score)) + "☆" * (5 - int(round(star_score)))
    print(f"Görsel: {bar}")
    print("-" * 60)