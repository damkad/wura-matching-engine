import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from geopy.distance import geodesic

# Load transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data(items_file='data/items.json', users_file='data/users.json'):
    with open(items_file) as f:
        items = json.load(f)
    with open(users_file) as f:
        users = json.load(f)
    return items, users

def haversine_score(user_loc, item_loc):
    distance_km = geodesic(user_loc, item_loc).km
    return np.exp(-distance_km / 20)  # Penalize based on distance

def trust_score(rating, max_rating=5.0):
    return rating / max_rating  # Normalize

def match_items_to_user(user, items, top_k=5):
    user_embedding = model.encode(user["needs"], convert_to_tensor=True)
    user_location = tuple(user["location"])

    scores = []
    for item in items:
        item_text = item["title"] + " " + item["description"]
        item_embedding = model.encode(item_text, convert_to_tensor=True)
        semantic_score = float(util.cos_sim(user_embedding, item_embedding))

        distance_weight = haversine_score(user_location, tuple(item["location"]))
        rating_weight = trust_score(item.get("owner_rating", 3.0))

        # Combine scores with adjustable weights
        combined_score = 0.6 * semantic_score + 0.25 * distance_weight + 0.15 * rating_weight
        scores.append((combined_score, item))

    # Sort by highest combined score
    top_matches = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
    return [match[1] for match in top_matches]


if __name__ == "__main__":
    items, users = load_data()
    user = users[0]  # Just test with the first user
    matches = match_items_to_user(user, items)

    print(f"\nTop matches for user '{user['name']}' ({user['location']}):\n")
    for match in matches:
        print(f"🏷 {match['title']}\n📍 {match['location']}\n📝 {match['description']}\n---")
