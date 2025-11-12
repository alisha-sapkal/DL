import numpy as np
import gensim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Lambda, Dense
import tensorflow.keras.backend as K

text = """Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence
concerned with the interactions between computers and human language, in particular how to program computers to
process and analyze large amounts of natural language data. The goal is a computer capable of "understanding"
the contents of documents, including the contextual nuances of the language within them. The technology can
then accurately extract information and insights contained in the documents as well as categorize and
organize the documents themselves."""


# Tokenize words from the paragraph
tokenized_words = [gensim.utils.simple_preprocess(sentence) for sentence in text.split('\n')]

# Gensim's tokenizer for the entire text
tokenizer = Tokenizer()

tokenizer.fit_on_texts(tokenized_words)
word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}

total_vocab = len(word_index) + 1
total_sentences = len(tokenized_words)

print(f"Total number of unique words: {total_vocab}")
print(f"Total number of sentences: {total_sentences}")

window_size = 2
X = []
y = []

for sentence in tokenized_words:
    for i, target_word in enumerate(sentence):
        context_words = []
        for j in range(i - window_size, i + window_size + 1):
            if j != i and j >= 0 and j < len(sentence):
                context_words.append(sentence[j])
        if context_words:
            X.append(context_words)
            y.append(target_word)

# Convert words to their integer indices
X_indexed = [[word_index[word] for word in context] for context in X]
y_indexed = [word_index[word] for word in y]

# Pad sequences to have the same length
X_padded = pad_sequences(X_indexed, maxlen=window_size*2)

# Convert target to one-hot encoding
y_categorical = to_categorical(y_indexed, num_classes=total_vocab)


embedding_dim = 100

model = Sequential([
    Embedding(input_dim=total_vocab, output_dim=embedding_dim, input_length=window_size*2),
    Lambda(lambda x: K.mean(x, axis=1), output_shape=(embedding_dim,)),
    Dense(total_vocab, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()


vect_file_path = 'vectors.txt'
# The number of vectors we will write is the number of words in our word_index.
num_vectors_to_write = len(word_index)
embedding_dim = 100 # Make sure this matches the embedding_dim from Cell 6

with open(vect_file_path, 'w') as vect_file:
    # Write the correct header: the actual number of vectors and their dimension.
    vect_file.write(f'{num_vectors_to_write} {embedding_dim}\n')

    weights = model.get_weights()[0]

    # Iterate through the word_index to write each word and its corresponding vector.
    for word, i in word_index.items():
        final_vec = ' '.join(map(str, list(weights[i, :])))
        vect_file.write(f'{word} {final_vec}\n')


# Choose a word from your vocabulary to get similar words
your_word = 'language'
try:
    similar_words = cbow_output.most_similar(positive=[your_word])
    print(f"Words similar to '{your_word}':")
    for word, similarity in similar_words:
        print(f"- {word}: {similarity:.4f}")
except KeyError:
    print(f"The word '{your_word}' is not in the vocabulary.")