class HybridRetriever:

    def __init__(self, bm25, vector_store, embedding_model):

        self.bm25 = bm25
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def search(self, query, k=5):

        bm25_results = self.bm25.search(query, k=10)

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        vector_results = self.vector_store.search(
            query_embedding,
            k=10
        )

        combined = {}

        for result in bm25_results:

            key = (
                result["source"],
                result["page"],
                result["text"]
            )

            combined[key] = {
                **result,
                "bm25_score": result["score"],
                "vector_score": 0
            }

        for result in vector_results:

            key = (
                result["source"],
                result["page"],
                result["text"]
            )

            if key not in combined:

                combined[key] = {
                    **result,
                    "bm25_score": 0,
                    "vector_score": result["score"]
                }

            else:

                combined[key]["vector_score"] = result["score"]

        results = list(combined.values())

        for result in results:

            result["hybrid_score"] = (
                0.5 * result["bm25_score"] +
                0.5 * result["vector_score"]
            )

        results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return results[:k]