# how we can seperatly taking 3 column and padding , concatenate

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM

# Initialize the tokenizer
tokenizer = Tokenizer()

# Tokenize the columns
column1_tokens = tokenizer.fit_on_texts(column1)
column2_tokens = tokenizer.fit_on_texts(column2)
column3_tokens = tokenizer.fit_on_texts(column3)

# Pad the sequences
max_length = max(len(column1_tokens), len(column2_tokens), len(column3_tokens))
column1_padded = pad_sequences(column1_tokens, maxlen=max_length)
column2_padded = pad_sequences(column2_tokens, maxlen=max_length)
column3_padded = pad_sequences(column3_tokens, maxlen=max_length)

# Concatenate the columns
concatenated_columns = np.concatenate((column1_padded, column2_padded, column3_padded), axis=1)

# Initialize the sequential model
model = Sequential()

# Add an embedding layer
model.add(Embedding(input_dim=vocab_size, output_dim=16, input_length=max_length))

# Add a LSTM layer
model.add(LSTM(32))

# Add a dense layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(concatenated_columns, labels, epochs=10, batch_size=32)
