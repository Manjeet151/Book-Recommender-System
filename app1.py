import pandas as pd
import pickle

# Load your CSV (the working file)
popular_df = pd.read_csv('popular.csv')  # Make sure this CSV works locally

# Save as a proper pickle
with open('popular.pkl', 'wb') as f:
    pickle.dump(popular_df, f)
