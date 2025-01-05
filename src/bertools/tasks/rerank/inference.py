from sentence_transformers.util import dot_score, semantic_search


def run_semantic_search(
    model,
    corpus: list[str],
    queries: list[str],
    top_k: int = 1,
    corpus_prefix: str = "",
    queries_prefix: str = "",
) -> list[dict[str, int | float | str]]:
    """
    This end-to-end semantic search function is tailored for small corpora,
    as the model is created and corpus and queries embedded on-the-fly.
    """
    # prepend prefixes to corpus and queries
    corpus_with_prefix = [corpus_prefix + t for t in corpus]
    queries_with_prefix = [queries_prefix + t for t in queries]

    # run pairwize dot products over normalized embeddings of queries and corpus
    sims = semantic_search(
        query_embeddings=model.encode(queries_with_prefix, normalize_embeddings=True),
        corpus_embeddings=model.encode(corpus_with_prefix, normalize_embeddings=True),
        top_k=top_k,
        score_function=dot_score,
    )
    # retrieve for each query the top most similars texts from corpus
    return [
        {
            "query": q,
            "article_index": pred["corpus_id"],
            "article": corpus[int(pred["corpus_id"])],
            "score": pred["score"],
            "rank": i + 1,
        }
        for q, top_k_classes in zip(queries, sims)
        for i, pred in enumerate(top_k_classes)
    ]
