import pandas as pd

def preprocess_data(df):
    # Convert Big/Small
    df['size'] = df['number'].apply(lambda x: 1 if x >= 5 else 0)

    # Encode colors
    color_map = {'red': 0, 'green': 1, 'violet': 2}
    df['color_encoded'] = df['color'].map(color_map)

    return df
