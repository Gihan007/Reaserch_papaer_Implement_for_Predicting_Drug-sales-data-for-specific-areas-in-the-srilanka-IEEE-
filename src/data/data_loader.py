import pandas as pd
import os

def load_category_data(category, base_path="."):
    """Load CSV for a given drug category (C1-C8)."""
    file_path = os.path.join(base_path, f"{category}.csv")
    df = pd.read_csv(file_path, parse_dates=["datum"], index_col="datum")
    return df

def load_all_categories(base_path="."):
    categories = [f"C{i}" for i in range(1, 9)]
    return {cat: load_category_data(cat, base_path) for cat in categories}
