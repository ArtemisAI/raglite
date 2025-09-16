"""Search and retrieve chunks."""

import contextlib
import re
import string
from collections import defaultdict
from itertools import groupby

import numpy as np
from langdetect import LangDetectException, detect
from sqlalchemy.orm import joinedload
from sqlmodel import Session, and_, col, func, or_, select, text

from raglite._config import RAGLiteConfig
from raglite._database import (
    Chunk,
    ChunkEmbedding,
    ChunkSpan,
    IndexMetadata,
    create_database_engine,
)
from raglite._embed import embed_strings
from raglite._typing import BasicSearchMethod, ChunkId, FloatVector, DistanceMetric


def _sqlite_vector_search(
    query_embedding: FloatVector,
    session: Session,
    num_results: int,
    oversample: int,
    config: RAGLiteConfig,
) -> tuple[list[ChunkId], list[float]]:
    """SQLite-specific vector search with sqlite-vec integration and PyNNDescent fallback."""
    import json
    import logging
    
    logger = logging.getLogger(__name__)
    engine = session.get_bind()
    sqlite_vec_available = getattr(engine, 'sqlite_vec_available', False)
    
    corrected_oversample = oversample * config.chunk_max_size / RAGLiteConfig.chunk_max_size
    num_hits = round(corrected_oversample) * max(num_results, 10)
    
    if sqlite_vec_available:
        # Use sqlite-vec for high-performance vector search
        try:
            return _sqlite_vec_search(query_embedding, session, num_results, num_hits, config)
        except Exception as e:
            logger.warning(f"sqlite-vec search failed, falling back to PyNNDescent: {e}")
    
    # Fallback to PyNNDescent for vector search
    return _pynndescent_search(query_embedding, session, num_results, num_hits, config)


def _sqlite_vec_search(
    query_embedding: FloatVector,
    session: Session,
    num_results: int,
    num_hits: int,
    config: RAGLiteConfig,
) -> tuple[list[ChunkId], list[float]]:
    """High-performance vector search using sqlite-vec extension."""
    import json
    
    # Convert query embedding to JSON for sqlite-vec
    query_json = json.dumps(query_embedding.tolist())
    
    # Map distance metrics to sqlite-vec functions
    metric_map = {
        "cosine": "cosine",
        "l2": "l2", 
        "dot": "dot"
    }
    
    if config.vector_search_distance_metric not in metric_map:
        raise ValueError(f"Distance metric {config.vector_search_distance_metric} not supported by sqlite-vec")
    
    metric = metric_map[config.vector_search_distance_metric]
    
    # Use sqlite-vec for vector similarity search
    statement = text(f"""
        SELECT ce.chunk_id, 
               vec_distance_{metric}(ce.embedding, :query_embedding) as distance
        FROM chunk_embedding ce
        ORDER BY distance
        LIMIT :limit
    """)
    
    try:
        results = session.execute(statement, {
            "query_embedding": query_json,
            "limit": num_hits
        }).fetchall()
        
        if not results:
            return [], []
        
        # Group by chunk_id and get the best similarity for each chunk
        chunk_scores = {}
        for chunk_id, distance in results:
            # Convert distance to similarity (higher is better)
            similarity = 1.0 - distance if config.vector_search_distance_metric != "dot" else distance
            chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, -float('inf')), similarity)
        
        # Sort by similarity and return top results
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:num_results]
        chunk_ids = [chunk_id for chunk_id, _ in sorted_chunks]
        similarities = [similarity for _, similarity in sorted_chunks]
        
        return chunk_ids, similarities
        
    except Exception as e:
        # If sqlite-vec functions aren't available, fall back to PyNNDescent
        raise RuntimeError(f"sqlite-vec search failed: {e}")


def _pynndescent_search(
    query_embedding: FloatVector,
    session: Session,
    num_results: int,
    num_hits: int,
    config: RAGLiteConfig,
) -> tuple[list[ChunkId], list[float]]:
    """Fallback vector search using PyNNDescent for compatibility."""
    import json
    from typing import cast
    
    try:
        import pynndescent
    except ImportError:
        raise ImportError("PyNNDescent is required for SQLite vector search fallback. Install with: pip install pynndescent")
    
    # Fetch all embeddings from database
    statement = select(ChunkEmbedding.chunk_id, ChunkEmbedding.embedding)
    db_results = session.exec(statement).all()
    
    if not db_results:
        return [], []
    
    # Extract embeddings and chunk IDs
    chunk_ids_all = [result.chunk_id for result in db_results]
    embeddings_all = np.vstack([result.embedding for result in db_results])
    
    # Build or load PyNNDescent index
    metric_map = {
        "cosine": "cosine",
        "l2": "euclidean",
        "dot": "dot"
    }
    
    metric = metric_map.get(config.vector_search_distance_metric, "cosine")
    
    # Create index with appropriate parameters for performance
    index = pynndescent.NNDescent(
        embeddings_all,
        metric=metric,
        n_neighbors=min(num_hits * 2, len(embeddings_all)),
        random_state=42
    )
    
    # Search for similar vectors
    indices, distances = index.query([query_embedding], k=min(num_hits, len(embeddings_all)))
    indices = indices[0]  # Extract first (and only) query result
    distances = distances[0]
    
    # Group by chunk_id and get the best similarity for each chunk
    chunk_scores = {}
    for idx, distance in zip(indices, distances):
        chunk_id = chunk_ids_all[idx]
        # Convert distance to similarity (higher is better)
        if metric == "dot":
            similarity = distance  # For dot product, higher is better
        else:
            similarity = 1.0 - distance  # For cosine and euclidean, lower distance is better
        
        chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, -float('inf')), similarity)
    
    # Sort by similarity and return top results
    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:num_results]
    chunk_ids_result = [chunk_id for chunk_id, _ in sorted_chunks]
    similarities = [similarity for _, similarity in sorted_chunks]
    
    return chunk_ids_result, similarities


def vector_search(
    query: str | FloatVector,
    *,
    num_results: int = 3,
    oversample: int = 4,
    config: RAGLiteConfig | None = None,
) -> tuple[list[ChunkId], list[float]]:
    """Search chunks using ANN vector search."""
    # Read the config.
    config = config or RAGLiteConfig()
    # Embed the query.
    query_embedding = (
        embed_strings([query], config=config)[0, :] if isinstance(query, str) else np.ravel(query)
    )
    # Apply the query adapter to the query embedding.
    if (
        config.vector_search_query_adapter
        and (Q := IndexMetadata.get("default", config=config).get("query_adapter")) is not None  # noqa: N806
    ):
        query_embedding = (Q @ query_embedding).astype(query_embedding.dtype)
    
    # Use database-specific vector search implementation
    with Session(create_database_engine(config)) as session:
        dialect = session.get_bind().dialect.name
        if dialect == "sqlite":
            # Use enhanced SQLite vector search
            return _sqlite_vector_search(query_embedding, session, num_results, oversample, config)
        else:
            # Use existing implementation for other databases
            corrected_oversample = oversample * config.chunk_max_size / RAGLiteConfig.chunk_max_size
            num_hits = round(corrected_oversample) * max(num_results, 10)
            dist = ChunkEmbedding.embedding.distance(  # type: ignore[attr-defined]
                query_embedding, metric=config.vector_search_distance_metric
            ).label("dist")
            sim = (1.0 - dist).label("sim")
            top_vectors = select(ChunkEmbedding.chunk_id, sim).order_by(dist).limit(num_hits).subquery()
            sim_norm = func.max(top_vectors.c.sim).label("sim_norm")
            statement = (
                select(top_vectors.c.chunk_id, sim_norm)
                .group_by(top_vectors.c.chunk_id)
                .order_by(sim_norm.desc())
                .limit(num_results)
            )
            rows = session.exec(statement).all()
            chunk_ids = [row[0] for row in rows]
            similarity = [float(row[1]) for row in rows]
            return chunk_ids, similarity


def keyword_search(
    query: str, *, num_results: int = 3, config: RAGLiteConfig | None = None
) -> tuple[list[ChunkId], list[float]]:
    """Search chunks using BM25 keyword search."""
    # Read the config.
    config = config or RAGLiteConfig()
    # Connect to the database.
    with Session(create_database_engine(config)) as session:
        dialect = session.get_bind().dialect.name
        if dialect == "postgresql":
            # Convert the query to a tsquery [1].
            # [1] https://www.postgresql.org/docs/current/textsearch-controls.html
            query_escaped = re.sub(f"[{re.escape(string.punctuation)}]", " ", query)
            tsv_query = " | ".join(query_escaped.split())
            # Perform keyword search with tsvector.
            statement = text("""
                SELECT id as chunk_id, ts_rank(to_tsvector('simple', body), to_tsquery('simple', :query)) AS score
                FROM chunk
                WHERE to_tsvector('simple', body) @@ to_tsquery('simple', :query)
                ORDER BY score DESC
                LIMIT :limit;
                """)
            results = session.execute(statement, params={"query": tsv_query, "limit": num_results})
        elif dialect == "duckdb":
            statement = text(
                """
                SELECT chunk_id, score
                FROM (
                    SELECT id AS chunk_id, fts_main_chunk.match_bm25(id, :query) AS score
                    FROM chunk
                ) sq
                WHERE score IS NOT NULL
                ORDER BY score DESC
                LIMIT :limit;
                """
            )
            results = session.execute(statement, params={"query": query, "limit": num_results})
        elif dialect == "sqlite":
            # Use FTS5 for keyword search in SQLite
            try:
                statement = text("""
                    SELECT chunk.id as chunk_id, bm25(chunk_fts) as score
                    FROM chunk_fts 
                    JOIN chunk ON chunk.id = chunk_fts.rowid
                    WHERE chunk_fts MATCH :query
                    ORDER BY score
                    LIMIT :limit
                """)
                results = session.execute(statement, params={"query": query, "limit": num_results})
            except Exception:
                # Fallback to simple LIKE search if FTS5 is not available
                statement = text("""
                    SELECT id as chunk_id, 1.0 as score
                    FROM chunk
                    WHERE body LIKE '%' || :query || '%'
                    ORDER BY id
                    LIMIT :limit
                """)
                results = session.execute(statement, params={"query": query, "limit": num_results})
        else:
            error_message = f"Keyword search is not supported for {dialect}."
            raise ValueError(error_message)
        # Unpack the results.
        results = list(results)  # type: ignore[assignment]
        chunk_ids = [result.chunk_id for result in results]
        keyword_score = [result.score for result in results]
    return chunk_ids, keyword_score


def reciprocal_rank_fusion(
    rankings: list[list[ChunkId]], *, k: int = 60, weights: list[float] | None = None
) -> tuple[list[ChunkId], list[float]]:
    """Reciprocal Rank Fusion."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        error = "The number of weights must match the number of rankings."
        raise ValueError(error)
    # Compute the RRF score.
    chunk_id_score: defaultdict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights, strict=True):
        for i, chunk_id in enumerate(ranking):
            chunk_id_score[chunk_id] += weight / (k + i)
    # Exit early if there are no results to fuse.
    if not chunk_id_score:
        return [], []
    # Rank RRF results according to descending RRF score.
    rrf_chunk_ids, rrf_score = zip(
        *sorted(chunk_id_score.items(), key=lambda x: x[1], reverse=True), strict=True
    )
    return list(rrf_chunk_ids), list(rrf_score)


def hybrid_search(  # noqa: PLR0913
    query: str,
    *,
    num_results: int = 3,
    oversample: int = 2,
    vector_search_weight: float = 0.75,
    keyword_search_weight: float = 0.25,
    config: RAGLiteConfig | None = None,
) -> tuple[list[ChunkId], list[float]]:
    """Search chunks by combining ANN vector search with BM25 keyword search."""
    config = config or RAGLiteConfig()
    
    # Check if we're using SQLite and can optimize the hybrid search
    with Session(create_database_engine(config)) as session:
        dialect = session.get_bind().dialect.name
        if dialect == "sqlite":
            return _sqlite_hybrid_search(
                query, session, num_results, oversample, 
                vector_search_weight, keyword_search_weight, config
            )
    
    # Use standard hybrid search for other databases
    # Run both searches.
    vs_chunk_ids, _ = vector_search(query, num_results=oversample * num_results, config=config)
    ks_chunk_ids, _ = keyword_search(query, num_results=oversample * num_results, config=config)
    # Combine the results with Reciprocal Rank Fusion (RRF).
    chunk_ids, hybrid_score = reciprocal_rank_fusion(
        [vs_chunk_ids, ks_chunk_ids], weights=[vector_search_weight, keyword_search_weight]
    )
    chunk_ids, hybrid_score = chunk_ids[:num_results], hybrid_score[:num_results]
    return chunk_ids, hybrid_score


def _sqlite_hybrid_search(
    query: str,
    session: Session,
    num_results: int,
    oversample: int,
    vector_search_weight: float,
    keyword_search_weight: float,
    config: RAGLiteConfig,
) -> tuple[list[ChunkId], list[float]]:
    """Optimized hybrid search specifically for SQLite with performance enhancements."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Get query embedding for vector search
    from raglite._embed import embed_strings
    query_embedding = embed_strings([query], config=config)[0, :]
    
    # Apply the query adapter to the query embedding if configured
    if (
        config.vector_search_query_adapter
        and (Q := IndexMetadata.get("default", config=config).get("query_adapter")) is not None  # noqa: N806
    ):
        query_embedding = (Q @ query_embedding).astype(query_embedding.dtype)
    
    expanded_results = oversample * num_results
    
    # Run vector search
    try:
        vs_chunk_ids, vs_scores = _sqlite_vector_search(
            query_embedding, session, expanded_results, oversample, config
        )
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")
        vs_chunk_ids, vs_scores = [], []
    
    # Run keyword search  
    try:
        ks_chunk_ids, ks_scores = _sqlite_keyword_search(
            query, session, expanded_results, config
        )
    except Exception as e:
        logger.warning(f"Keyword search failed: {e}")
        ks_chunk_ids, ks_scores = [], []
    
    # Combine the results with Reciprocal Rank Fusion (RRF)
    chunk_ids, hybrid_score = reciprocal_rank_fusion(
        [vs_chunk_ids, ks_chunk_ids], 
        weights=[vector_search_weight, keyword_search_weight]
    )
    
    # Return top results
    return chunk_ids[:num_results], hybrid_score[:num_results]


def _sqlite_keyword_search(
    query: str,
    session: Session,
    num_results: int,
    config: RAGLiteConfig,
) -> tuple[list[ChunkId], list[float]]:
    """SQLite-specific keyword search with FTS5 optimization."""
    # Use FTS5 for keyword search in SQLite
    try:
        statement = text("""
            SELECT chunk.id as chunk_id, bm25(chunk_fts) as score
            FROM chunk_fts 
            JOIN chunk ON chunk.id = chunk_fts.rowid
            WHERE chunk_fts MATCH :query
            ORDER BY score
            LIMIT :limit
        """)
        results = session.execute(statement, params={"query": query, "limit": num_results})
        results = list(results)
        chunk_ids = [result.chunk_id for result in results]
        keyword_score = [result.score for result in results]
        return chunk_ids, keyword_score
    except Exception:
        # Fallback to simple LIKE search if FTS5 is not available
        statement = text("""
            SELECT id as chunk_id, 1.0 as score
            FROM chunk
            WHERE body LIKE '%' || :query || '%'
            ORDER BY id
            LIMIT :limit
        """)
        results = session.execute(statement, params={"query": query, "limit": num_results})
        results = list(results)
        chunk_ids = [result.chunk_id for result in results]
        keyword_score = [result.score for result in results]
        return chunk_ids, keyword_score


def retrieve_chunks(
    chunk_ids: list[ChunkId], *, config: RAGLiteConfig | None = None
) -> list[Chunk]:
    """Retrieve chunks by their ids."""
    if not chunk_ids:
        return []
    with Session(create_database_engine(config := config or RAGLiteConfig())) as session:
        chunks = list(
            session.exec(
                select(Chunk)
                .where(col(Chunk.id).in_(chunk_ids))
                # Eagerly load chunk.document.
                .options(joinedload(Chunk.document))  # type: ignore[arg-type]
            ).all()
        )
    chunks = sorted(chunks, key=lambda chunk: chunk_ids.index(chunk.id))
    return chunks


def retrieve_chunk_spans(
    chunk_ids: list[ChunkId] | list[Chunk],
    *,
    neighbors: tuple[int, ...] | None = (-1, 1),
    config: RAGLiteConfig | None = None,
) -> list[ChunkSpan]:
    """Group chunks into contiguous chunk spans and retrieve them.

    Chunk spans are ordered according to the aggregate relevance of their underlying chunks, as
    determined by the order in which they are provided to this function.
    """
    # Exit early if the input is empty.
    if not chunk_ids:
        return []
    # Retrieve the chunks.
    config = config or RAGLiteConfig()
    chunks: list[Chunk] = (
        retrieve_chunks(chunk_ids, config=config)  # type: ignore[arg-type,assignment]
        if all(isinstance(chunk_id, ChunkId) for chunk_id in chunk_ids)
        else chunk_ids
    )
    # Assign a reciprocal ranking score to each chunk based on its position in the original list.
    chunk_id_to_score = {chunk.id: 1 / (i + 1) for i, chunk in enumerate(chunks)}
    # Extend the chunks with their neighbouring chunks.
    with Session(create_database_engine(config)) as session:
        if neighbors:
            neighbor_conditions = [
                and_(Chunk.document_id == chunk.document_id, Chunk.index == chunk.index + offset)
                for chunk in chunks
                for offset in neighbors
            ]
            chunks += list(
                session.exec(
                    select(Chunk)
                    .where(or_(*neighbor_conditions))
                    # Eagerly load chunk.document.
                    .options(joinedload(Chunk.document))  # type: ignore[arg-type]
                ).all()
            )
    # Deduplicate and sort the chunks by document_id and index (needed for groupby).
    unique_chunks = sorted(set(chunks), key=lambda chunk: (chunk.document_id, chunk.index))
    # Group the chunks into contiguous segments.
    chunk_spans: list[ChunkSpan] = []
    for _, group in groupby(unique_chunks, key=lambda chunk: chunk.document_id):
        chunk_sequence: list[Chunk] = []
        for chunk in group:
            if not chunk_sequence or chunk.index == chunk_sequence[-1].index + 1:
                chunk_sequence.append(chunk)
            else:
                chunk_spans.append(ChunkSpan(chunks=chunk_sequence))
                chunk_sequence = [chunk]
        chunk_spans.append(ChunkSpan(chunks=chunk_sequence))
    # Rank segments according to the aggregate relevance of their chunks.
    chunk_spans.sort(
        key=lambda chunk_span: sum(
            chunk_id_to_score.get(chunk.id, 0.0) for chunk in chunk_span.chunks
        ),
        reverse=True,
    )
    return chunk_spans


def rerank_chunks(
    query: str, chunk_ids: list[ChunkId] | list[Chunk], *, config: RAGLiteConfig | None = None
) -> list[Chunk]:
    """Rerank chunks according to their relevance to a given query."""
    # Retrieve the chunks.
    config = config or RAGLiteConfig()
    chunks: list[Chunk] = (
        retrieve_chunks(chunk_ids, config=config)  # type: ignore[arg-type,assignment]
        if all(isinstance(chunk_id, ChunkId) for chunk_id in chunk_ids)
        else chunk_ids
    )
    # Exit early if no reranker is configured or if the input is empty.
    if not config.reranker or not chunks:
        return chunks
    # Select the reranker.
    if isinstance(config.reranker, dict):
        # Detect the languages of the chunks and queries.
        with contextlib.suppress(LangDetectException):
            langs = {detect(str(chunk)) for chunk in chunks}
            langs.add(detect(query))
        # If all chunks and the query are in the same language, use a language-specific reranker.
        rerankers = config.reranker
        if len(langs) == 1 and (lang := next(iter(langs))) in rerankers:
            reranker = rerankers[lang]
        else:
            reranker = rerankers.get("other")
    else:
        # A specific reranker was configured.
        reranker = config.reranker
    # Rerank the chunks.
    if reranker:
        results = reranker.rank(query=query, docs=[str(chunk) for chunk in chunks])
        chunks = [chunks[result.doc_id] for result in results.results]
    return chunks


def search_and_rerank_chunks(
    query: str,
    *,
    num_results: int = 8,
    oversample: int = 4,
    search: BasicSearchMethod = hybrid_search,
    config: RAGLiteConfig | None = None,
) -> list[Chunk]:
    """Search and rerank chunks."""
    chunk_ids, _ = search(query, num_results=oversample * num_results, config=config)
    chunks = rerank_chunks(query, chunk_ids, config=config)[:num_results]
    return chunks


def search_and_rerank_chunk_spans(  # noqa: PLR0913
    query: str,
    *,
    num_results: int = 8,
    oversample: int = 4,
    neighbors: tuple[int, ...] | None = (-1, 1),
    search: BasicSearchMethod = hybrid_search,
    config: RAGLiteConfig | None = None,
) -> list[ChunkSpan]:
    """Search and rerank chunks, and then collate into chunk spans."""
    chunk_ids, _ = search(query, num_results=oversample * num_results, config=config)
    chunks = rerank_chunks(query, chunk_ids, config=config)[:num_results]
    chunk_spans = retrieve_chunk_spans(chunks, neighbors=neighbors, config=config)
    return chunk_spans
