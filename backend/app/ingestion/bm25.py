from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, documents):

        self.documents = documents

        tokenized = [
            doc["text"].lower().split()
            for doc in documents
        ]

        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, k=5):

        scores = self.bm25.get_scores(
            query.lower().split()
        )

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for index, score in ranked[:k]:

            result = self.documents[index].copy()

            result["score"] = float(score)

            results.append(result)

        return results