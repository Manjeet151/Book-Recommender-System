from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from fuzzywuzzy import process
import time
import os 
import pickle

app = Flask(__name__)

# Load data once at startup
print("Loading data files...")
start_time = time.time()

try:
    # Try loading from CSV files first
    popular_df = pd.read_csv('popular.csv')
    pt = pd.read_csv('pt.csv', index_col=0)
    books = pd.read_csv('books.csv')
    similarity_scores = np.load('similarity_scores.npy')
    print("✓ Loaded from CSV files")
    
except FileNotFoundError:
    # Fallback to pickle files if CSV doesn't exist
    print("CSV files not found, loading from pickle...")
    try:
        popular_df = pickle.load(open('popular.pkl', 'rb'))
        pt = pickle.load(open('pt.pkl', 'rb'))
        books = pickle.load(open('books.pkl', 'rb'))
        similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))
        print("✓ Loaded from pickle files")
        
        # Auto-convert to CSV for next time
        popular_df.to_csv('popular.csv', index=False)
        pt.to_csv('pt.csv', index=True)
        books.to_csv('books.csv', index=False)
        np.save('similarity_scores.npy', similarity_scores)
        print("✓ Auto-converted to CSV for next time")
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        raise

# Precompute book dictionary for fast lookups
book_dict = {}
for _, row in books.iterrows():
    if row['Book-Title'] not in book_dict:
        book_dict[row['Book-Title']] = {
            'author': row['Book-Author'],
            'image': row['Image-URL-M']
        }

# Precompute all books list for fuzzy matching
all_books = pt.index.tolist()

print(f"Data loaded in {time.time() - start_time:.2f} seconds")

@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend():
    start_time = time.time()
    user_input = request.form.get('user_input')
    
    # Fuzzy matching with limit to improve performance
    matched_title, score = process.extractOne(user_input, all_books)
    
    if score < 50:
        print(f"Fuzzy match failed (score: {score}) in {time.time() - start_time:.2f}s")
        return render_template('recommend.html', 
                               error="Book not found. Try another title!",
                               user_input=user_input)
    
    # Get index of matched book
    index = np.where(pt.index == matched_title)[0][0]
    
    # Get similar books (limit to top 5 for performance)
    similar_items = sorted(
        list(enumerate(similarity_scores[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]
    
    # Collect recommendations using precomputed dictionary
    data = []
    for i in similar_items:
        book_title = pt.index[i[0]]
        if book_title in book_dict:
            book = book_dict[book_title]
            item = [
                book_title,
                book['author'],
                book['image']
            ]
            data.append(item)
    
    processing_time = time.time() - start_time
    print(f"Request processed in {processing_time:.2f} seconds")
    
    if not data:
        return render_template('recommend.html', 
                               error="No recommendations found!",
                               user_input=user_input,
                               matched=matched_title)
    
    return render_template('recommend.html', 
                           data=data, 
                           user_input=user_input,
                           matched=matched_title)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)