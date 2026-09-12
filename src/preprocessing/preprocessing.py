from .cleaning import clean_data


def preprocess_data(df):
    """Run the complete Layer 2 preprocessing pipeline."""

    result = clean_data(df)

    return result