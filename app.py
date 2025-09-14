from flask import Flask, render_template, request
import pickle
import numpy as np
from fuzzywuzzy import process
import time
import os

app = Flask(__name__)

# Use absolute paths to make sure Render finds the files
base_dir = os.path.dirname(__file__)

def load_pickle(filename):
    with open(os.path.join(base_dir, filename), 'rb') as f:
        return pickle.load(f)

print("Loading data files...")
start_time = time.time()

popular_df = load_pickle('popular.pkl')
pt = load_pickle('pt.pkl')
books = load_pickle('books.pkl')
similarity_scores = load_pickle('similarity_scores.pkl')

# Precompute book dictionary for fast lookups
book_dict = {row['Book-Title']: {'author': row['Book-Author'], 'image': row['Image-URL-M']}
             for _, row in books.iterrows()}

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
    app.run(debug=True)
