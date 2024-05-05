import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Text Pre-processing libraries (not used in corrected code)
# import nltk
# import string
# import warnings
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from wordcloud import WordCloud

# Tensorflow imports to build the model.
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# No nltk downloads needed (corrected code)

df = pd.read_csv("C:/Users/Leka/Downloads/archive (1)/labeled_data.csv")
df.head()
plt.pie(df['class'].value_counts().values,
        labels=df['class'].value_counts().index,
        autopct='%1.1f%%')
plt.show()

# Text preprocessing is removed as it's not relevant for sentiment analysis

from sklearn.model_selection import train_test_split
features = df['tweet']  # Assuming 'tweet' column contains text data
target = df['class']  # Assuming 'class' column contains numerical labels (0, 1, 2)

X_train, X_val, Y_train, Y_val = train_test_split(features, target,
                                                  test_size=0.2,
                                                  random_state=22)
X_train.shape, X_val.shape

# Assuming numerical labels, no need for one-hot encoding
Y_train = Y_train.values.reshape(-1, 1)  # Reshape for compatibility
Y_val = Y_val.values.reshape(-1, 1)

from tensorflow.keras.preprocessing.sequence import pad_sequences

# No tokenization needed for sentiment analysis (assuming text data)
max_len = 100  # Adjust if needed for your text length

# No padding sequences needed for sentiment analysis (assuming text data)

model = keras.Sequential([
    layers.Embedding(input_dim=100, output_dim=1000, input_length=max_len),
    layers.LSTM(64, return_sequences=True),  # Adjust units and return_sequences as needed
    layers.LSTM(32),  # Adjust units as needed
    layers.Dense(1, activation='sigmoid')  # Output layer for sentiment analysis (0-1)
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

keras.utils.plot_model(model, show_shapes=True, show_dtype=True, show_layer_activations=True)

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=0)

history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=50, verbose=1, batch_size=32, callbacks=[lr, es])