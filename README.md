
#  Wura Matching Engine

A geolocation-aware, trust-weighted, and AI-powered matching system that connects donated items with users in need — built for **Wura**, a mobile-first platform where people can give away unused items and women and families can request what they need privately and safely.

---

##  Features

-  **Semantic Matching** using state-of-the-art transformer embeddings (`sentence-transformers`)
-  **Location Awareness** with distance-based scoring (Haversine formula)
-  **Trust Scores** using item owner ratings
-  **Custom Scoring Algorithm** combining relevance, proximity, and reliability
-  Designed to be **fully open source and extensible**

---

## 📁 Project Structure

```
.
├── match_engine.py         # Main matching logic
├── data/
│   ├── items.json          # Donated items with metadata
│   └── users.json          # Users and their needs
├── requirements.txt        # Python dependencies
└── README.md               # You're here
```

---

## 🔧 Installation

```bash
git clone https://github.com/yourusername/wura-matching-engine.git
cd wura-matching-engine

pip install -r requirements.txt
```

**requirements.txt**
```
sentence-transformers
geopy
numpy
```

---

## 🧪 Run the Demo

```bash
python match_engine.py
```

This will:
- Load users and items from the `data/` folder
- Run the AI-powered matching algorithm
- Print the top recommended items per user

---

##  Example Output

```
Top matches for 'Amina' ([6.5244, 3.3792]):

 Children’s School Shoes
 [6.52, 3.375]  ⭐ 4.8
 Gently used black leather school shoes for girls, size 2.
---
 Backpack with Zipper
 [6.523, 3.38]  ⭐ 4.5
 Durable school backpack with cartoon print, good for ages 5-10.
---
```

---

##  How It Works

1. **User Need** → Encoded with a Transformer model (`all-MiniLM-L6-v2`)
2. **Item Descriptions** → Also embedded
3. **Cosine Similarity** → Measures semantic match
4. **Distance Score** → Penalizes far items
5. **Rating Score** → Rewards high-trust donors
6. **Final Score** → Weighted combination of all three

```python
final_score = 0.6 * semantic_score + 0.25 * distance_score + 0.15 * trust_score
```

---

##  Why It Matters

Wura’s matching engine ensures that:
- The **right help** gets to the **right people**
- **Proximity** and **trust** are prioritized
- Matching is **scalable**, **safe**, and **impactful**

---

## 👥 Contributing

Pull requests are welcome! If you have ideas for improving the algorithm or expanding the system (e.g. categories, images, feedback loops), we’d love to collaborate.

---

## 🔒 License

MIT — Use it, build on it, improve it.
