import faiss
import numpy as np


class VectorStore:

    def __init__(self):
        self.index = None
        self.documents = []

    def build(self, embeddings, documents):

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(
            np.array(embeddings).astype("float32")
        )

        self.documents = documents

    def search(self, query_embedding, k=5):

        scores, indices = self.index.search(
            np.array(query_embedding).astype("float32"),
            k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            result = self.documents[index].copy()

            result["score"] = float(score)

            results.append(result)

        return results