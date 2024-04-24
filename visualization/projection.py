import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors

# Load pre-trained GloVe word embeddings
glove_model = KeyedVectors.load_word2vec_format('path_to_glove_vectors.txt', binary=False)

# Select a few words for visualization
words = ['king', 'queen', 'man', 'woman', 'apple', 'orange', 'car', 'bike']

# Get the embeddings for the selected words
word_vectors = np.array([glove_model[word] for word in words])

# Apply PCA to reduce dimensionality to 2D
pca = PCA(n_components=2)
word_embeddings_2d = pca.fit_transform(word_vectors)

# Plot the word embeddings
plt.figure(figsize=(8, 6))
plt.scatter(word_embeddings_2d[:, 0], word_embeddings_2d[:, 1], marker='o')

# Annotate each point with the corresponding word
for i, word in enumerate(words):
    plt.annotate(word, xy=(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1]))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Word Embeddings Visualization (GloVe)')
plt.grid(True)
plt.show()

