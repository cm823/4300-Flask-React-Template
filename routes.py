import json
from flask import send_from_directory, request, jsonify
from models import db, Episode, Review
import os

# Search function
def json_search(query):
    # Query episodes with matching title and join with reviews
    results = db.session.query(Episode, Review).join(
        Review, Episode.id == Review.id
    ).filter(
        Episode.title.ilike(f'%{query}%')
    ).all()
    
    # Convert results to JSON format
    matches = []
    for episode, review in results:
        matches.append({
            'title': episode.title,
            'descr': episode.descr,
            'imdb_rating': review.imdb_rating
        })
    
    return matches

def register_routes(app):
    """Register all routes with the Flask app"""
    
    # Serve React App
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, 'index.html')

    # API Routes
    @app.route("/api/episodes")
    def episodes_search():
        text = request.args.get("title", "")
        results = json_search(text)
        return jsonify(results)
