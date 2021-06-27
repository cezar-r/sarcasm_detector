import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000 
embedding_dim = 16
max_length = 100 
trunc_type = 'post'
padding_type = 'post'
oov_token = '<OOV>'
training_size = 20000 

def main():
	data = []
	with open("../data/sarcasm.json", "r") as f:
		for line in f:
			data.append(json.loads(line))

	sentences = []
	labels = []

	for item in data:
		sentences.append(item['headline'])
		labels.append(item['is_sarcastic'])

	train_sentences = sentences[0:training_size]
	test_sentences = sentences[training_size:]

	train_labels = labels[0:training_size]
	test_labels = labels[training_size:]

	tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_token)
	tokenizer.fit_on_texts(train_sentences)

	train_sequences = tokenizer.texts_to_sequences(train_sentences)
	train_padded = pad_sequences(train_sequences, maxlen = max_length, padding=padding_type, truncating=trunc_type)

	test_sequences = tokenizer.texts_to_sequences(test_sentences)
	test_padded = pad_sequences(test_sequences, maxlen = max_length, padding=padding_type, truncating=trunc_type)

	train_padded = np.array(train_padded)
	train_labels = np.array(train_labels)
	test_padded = np.array(test_padded)
	test_labels = np.array(test_labels)

	model = tf.keras.Sequential([
		tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
		tf.keras.layers.GlobalAveragePooling1D(),
		tf.keras.layers.Dense(24, activation='relu'),
		tf.keras.layers.Dense(1, activation='sigmoid')
	])

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	# model.summary()

	num_epochs = 30

	history = model.fit(train_padded, train_labels, epochs = num_epochs, validation_data = (test_padded, test_labels))

	return model, tokenizer


def get_input():

	sentence = input("\nEnter sentence:\n")
	return [sentence]


def predict(model, tokenizer):
	sentence = get_input()

	sequences = tokenizer.texts_to_sequences(sentence)

	padded = pad_sequences(sequences, maxlen = max_length, padding = padding_type, truncating=trunc_type)

	score = model.predict(padded)[0][0] * 10
	print('')
	print(f'Sarcasm score for sentence: "{sentence[0]}"')
	print(round(score,2), ' / 10')

	if score < 1:
		print('Not sarcastic at all')
	elif score < 4:
		print('Not very sarcastic')
	elif score < 6:
		print('Neutral')
	elif score < 9:
		print('Pretty sarcastic')
	else:
		print('Very Sarcastic')

def plot_graph(history, string):
	plt.plot(history.history[string])
	plt.plot(history.history['val_'+string])
	plt.xlabel("Epochs")
	plt.ylabel(string)
	plt.legend([string, 'val_'+string])
	plt.show()

# plot_graph(history, "accuracy")
# plot_graph(history, "loss")

if __name__ == '__main__':
	model, tokenizer = main()
	predict(model, tokenizer)