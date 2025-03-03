import faiss
import numpy as np

# Create some test data
dimension = 128  # Dimension of your vectors
nb_vectors = 100  # Number of vectors
np.random.seed(42)
vectors = np.random.random((nb_vectors, dimension)).astype('float32')

# Create a simple index
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

# Test search
k = 5  # Number of nearest neighbors to retrieve
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k)

print(f"Test search results - indices: {indices}, distances: {distances}")
print("FAISS installation successful!")