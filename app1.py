import pandas as pd
import pickle

# Load your CSV (or dataframe that works locally)
popular_df = pd.read_csv('popular.csv')  # replace with your original CSV if needed

# Save it as a pickle
with open('popular.pkl', 'wb') as f:
    pickle.dump(popular_df, f)
