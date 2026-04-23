"""
LLM chat route — only loaded when USE_LLM = True in routes.py.
Adds a POST /api/chat endpoint that performs LLM-driven RAG.

Setup:
  1. Add API_KEY=your_key to .env
  2. Set USE_LLM = True in routes.py
"""
import json
import os
import re
import logging
from flask import request, jsonify, Response, stream_with_context
from infosci_spark_client import LLMClient

logger = logging.getLogger(__name__)


def llm_search_decision(client, user_message):
    """Ask the LLM whether to search the DB and which word to use."""
    messages = [
        {
            "role": "system",
            "content": (
                "You have access to a database of Keeping Up with the Kardashians episode titles, "
                "descriptions, and IMDB ratings. Search is by a single word in the episode title. "
                "Reply with exactly: YES followed by one space and ONE word to search (e.g. YES wedding), "
                "or NO if the question does not need episode data."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    response = client.chat(messages)
    content = (response.get("content") or "").strip().upper()
    logger.info(f"LLM search decision: {content}")
    if re.search(r"\bNO\b", content) and not re.search(r"\bYES\b", content):
        return False, None
    yes_match = re.search(r"\bYES\s+(\w+)", content)
    if yes_match:
        return True, yes_match.group(1).lower()
    if re.search(r"\bYES\b", content):
        return True, "Kardashian"
    return False, None


def rewrite_query(client, original_query):
    """Ask the LLM to expand the query with synonyms for better retrieval."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a search query optimizer for a Wikipedia discovery engine. "
                "Given a user query, do the following:\n"
                "1. Identify the most important keywords in the query.\n"
                "2. Find meaningful synonyms or related terms for each keyword.\n"
                "3. Return the original query with those synonyms added to it.\n"
                "Return only the expanded query — no explanation, no bullet points, no punctuation at the end."
            ),
        },
        {"role": "user", "content": original_query},
    ]
    response = client.chat(messages)
    rewritten = (response.get("content") or "").strip()
    return rewritten if rewritten else original_query


def generate_summary(client, original_query, articles):
    """Ask the LLM to summarize why the retrieved articles are relevant."""
    titles_text = "\n".join(f"- {a['title']}" for a in articles)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for a Wikipedia discovery engine. "
                "Given a user's query and a list of retrieved Wikipedia articles, "
                "write 2-3 sentences explaining why these articles are relevant to the query "
                "and what interesting connections exist between them. Be engaging and concise."
            ),
        },
        {
            "role": "user",
            "content": f'User query: "{original_query}"\n\nRetrieved articles:\n{titles_text}',
        },
    ]
    response = client.chat(messages)
    return (response.get("content") or "").strip()


def register_rag_route(app, run_pipeline):
    """Register POST /api/rag_query. run_pipeline(query, scoring_mode, path_length) -> branches."""

    @app.route("/api/rag_query", methods=["POST"])
    def rag_query():
        data = request.get_json() or {}
        original_query = (data.get("article") or "").strip()
        scoring_mode = data.get("scoring_mode", "tfidf")
        path_length = int(data.get("path_length", 5))

        if not original_query:
            return jsonify({"error": "article is required"}), 400

        modified_query = original_query
        summary = ""

        # Step 1: rewrite the query (best-effort — fall back to original on failure)
        api_key = os.getenv("API_KEY")
        client = LLMClient(api_key=api_key) if api_key else None

        if client:
            try:
                modified_query = rewrite_query(client, original_query)
                logger.info(f"RAG query rewrite: '{original_query}' -> '{modified_query}'")
            except Exception as e:
                logger.error(f"Query rewrite failed: {e}")
                modified_query = original_query

        # Step 2: run the IR pipeline with the modified query
        try:
            branches = run_pipeline(modified_query, scoring_mode, path_length)
        except Exception as e:
            logger.error(f"IR pipeline failed: {e}")
            return jsonify({"error": f"IR pipeline error: {e}"}), 500

        # Step 3: generate a summary (best-effort — skip on failure)
        if client:
            all_articles = [node for branch in branches for node in branch]
            if all_articles:
                try:
                    summary = generate_summary(client, original_query, all_articles)
                except Exception as e:
                    logger.error(f"Summary generation failed: {e}")

        return jsonify({
            "original_query": original_query,
            "modified_query": modified_query,
            "summary": summary,
            "results": branches,
        })


def register_chat_route(app, json_search):
    """Register the /api/chat SSE endpoint. Called from routes.py."""

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        api_key = os.getenv("API_KEY")
        if not api_key:
            return jsonify({"error": "API_KEY not set — add it to your .env file"}), 500

        client = LLMClient(api_key=api_key)
        use_search, search_term = llm_search_decision(client, user_message)

        if use_search:
            episodes = json_search(search_term or "Kardashian")
            context_text = "\n\n---\n\n".join(
                f"Title: {ep['title']}\nDescription: {ep['descr']}\nIMDB Rating: {ep['imdb_rating']}"
                for ep in episodes
            ) or "No matching episodes found."
            messages = [
                {"role": "system", "content": "Answer questions about Keeping Up with the Kardashians using only the episode information provided."},
                {"role": "user", "content": f"Episode information:\n\n{context_text}\n\nUser question: {user_message}"},
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant for Keeping Up with the Kardashians questions."},
                {"role": "user", "content": user_message},
            ]

        def generate():
            if use_search and search_term:
                yield f"data: {json.dumps({'search_term': search_term})}\n\n"
            try:
                for chunk in client.chat(messages, stream=True):
                    if chunk.get("content"):
                        yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': 'Streaming error occurred'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
