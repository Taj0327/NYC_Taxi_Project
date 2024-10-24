import pandas as pd
from sklearn.model_selection import train_test_split

# data loading
ytd_2024_01_df = pd.read_parquet('data/yellow_tripdata_2024-01.parquet')

def raw_taxi_df(filename: str) -> pd.DataFrame:
    """
    Load raw taxi dataframe from parquet
    """

    return pd.read_parquet(path=filename)

# data cleaning
def clean_taxi_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a clean taxi DataFrame that throws out non-numerical or outlying numerical values.
    """

    # drop nans
    clean_df = raw_df.dropna()

    # remove trips longer than 100
    clean_df = clean_df[clean_df["trip_distance"] < 100]

    # add columns for travel time deltas and time minutes
    clean_df["time_deltas"] = clean_df["tpep_dropoff_datetime"] - clean_df["tpep_pickup_datetime"]
    clean_df["time_mins"] = pd.to_numeric(clean_df["time_deltas"]) / 6**10

    return clean_df

def split_taxi_data(clean_df: pd.DataFrame, 
                    x_columns: list[str], 
                    y_column: str, 
                    train_size: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split an x, y dataset selected from a clean dataframe; return x_train, y_train, x_test, y_test
    Args:
        clean_df(pd.DataFrame): Cleaned data.
        x_columns(list[str]): The names of the target columns.
        y_column(str): The name of the target feature to be predicted.
        train_size(int): The absolute number of samples to be included in the training set.
    """

    return train_test_split(clean_df[x_columns], clean_df[[y_column]], train_size=train_size)

