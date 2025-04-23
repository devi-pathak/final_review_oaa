# preprocess_and_save.py
import pandas as pd
import os
import warnings

# Check if preprocessor.py exists
if not os.path.exists('preprocessor.py'):
    raise FileNotFoundError("preprocessor.py not found in the current directory!")

# Import preprocess from preprocessor.py
from preprocessor import preprocess

# Define region_df at module level for use in app.py
region_df = pd.read_csv('noc_regions.csv', dtype={'NOC': 'category', 'region': 'category'})

def preprocess_and_save_data():
    """
    Load, preprocess, aggregate, and save Olympic data for medal prediction.
    Generates agg_df.csv with corrected features.
    """
    # Load with optimized dtypes
    dtypes = {
        'ID': 'int32', 'Age': 'float32', 'Height': 'float32', 'Weight': 'float32',
        'NOC': 'category', 'Year': 'int32', 'Season': 'category', 'Medal': 'category'
    }
    try:
        df = pd.read_csv('athlete_events.csv', dtype=dtypes)
        region_df = pd.read_csv('noc_regions.csv', dtype={'NOC': 'category', 'region': 'category'})
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found: {e}")

    # Preprocess using your function
    df = preprocess(df, region_df)

    # Map legacy NOCs to modern equivalents
    legacy_noc_map = {
        'URS': 'RUS',
        'EUN': 'RUS',
    }

    df['NOC'] = df['NOC'].replace(legacy_noc_map)

    # ------------------ Correct Medal Aggregation ------------------

    # 1. Keep only medal-winning entries
    medal_df = df[df['Medal'].notna()]

    # 2. Deduplicate Team/NOC/Year/Event/Medal (remove overcount of team medals)
    deduped_medal_df = medal_df.drop_duplicates(subset=['Team', 'NOC', 'Year', 'Event', 'Medal'])

    # 3. Group and aggregate
    agg_df = deduped_medal_df.groupby(['NOC', 'Year']).agg({
        'Medal': 'count',       # correct medal count
        'Age': 'mean',          # average age
        'Height': 'mean',       # average height
        'Weight': 'mean',       # average weight
        'ID': 'count'           # rough team size
    }).reset_index()

    agg_df.rename(columns={'Medal': 'Total_Medals'}, inplace=True)

    # 4. Add previous medals feature
    agg_df['Prev_Medals'] = agg_df.groupby('NOC')['Total_Medals'].shift(1).fillna(0)

    # 5. Handle missing Age/Height/Weight
    for col in ['Age', 'Height', 'Weight']:
        agg_df[col] = agg_df.groupby('NOC')[col].transform(lambda x: x.fillna(x.mean()))
        agg_df[col] = agg_df[col].fillna(agg_df[col].mean())

    # ----------------- Final Validations -----------------

    if agg_df.empty:
        raise ValueError("Aggregated DataFrame is empty. Check preprocessing steps.")

    # Save final CSV
    agg_df.to_csv('agg_df.csv', index=False)
    print("✅ Corrected aggregated data saved as 'agg_df.csv'.")

# -------- Main --------
if __name__ == "__main__":
    preprocess_and_save_data()