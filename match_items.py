import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_data(items_file='data/items.json', users_file='data/users.json'):
    with open(items_file) as f:
        items = json.load(f)
    with open(users_file) as f:
        users = json.load(f)
    return items, users

def match_items_to_user(user, items, top_k=5):
    """
    Matches a user's needs to available items using TF-IDF and cosine similarity.
    Returns top_k matched items.
    """
    user_query = user["needs"]
    item_texts = [item["title"] + " " + item["description"] for item in items]
    
    corpus = [user_query] + item_texts
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Get cosine similarities between user query and each item
    cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    top_indices = np.argsort(cosine_scores)[::-1][:top_k]

    return [items[i] for i in top_indices]

if __name__ == "__main__":
    items, users = load_data()
    user = users[0]  # Just test with the first user
    matches = match_items_to_user(user, items)

    print(f"\nTop matches for user '{user['name']}' ({user['location']}):\n")
    for match in matches:
        print(f"🏷 {match['title']}\n📍 {match['location']}\n📝 {match['description']}\n---")
