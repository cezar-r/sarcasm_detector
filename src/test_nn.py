from testing import main

model = main()


def get_input():

	sentence = input("Enter sentence:\n")
	return sentence


def predict():
	sentence = get_input()

	sequences = tokenizer.texts_to_sequences(sentence)

	padded = pad_sequences(sequences, maxlen = max_length, padding = padding_type, truncating=trunc_type)

	print(model.predict(padded))

predict()