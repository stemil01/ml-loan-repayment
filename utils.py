import numpy as np
import pandas as pd

# The `agg_numeric_columns` function creates a new DataFrame with aggregated
# numeric columns of the given DataFrame.
def agg_numeric_columns(df, groupby_columns, metrics=["count", "mean", "sum", "min", "max"], exclude_columns=None):
    exclude_columns = exclude_columns or []

    # Find all numeric columns that are not used for grouping
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    numeric_columns = [c for c in numeric_columns if c not in groupby_columns and c not in exclude_columns]

    if not numeric_columns:
        return df[groupby_columns].drop_duplicates().reset_index(drop=True)
    
    # Compute the aggregations and return a multi-index DataFrame
    multi_index_agg = df.groupby(groupby_columns)[numeric_columns].agg(metrics)

    # Flatten the DataFrame
    agg_df = pd.DataFrame({
        f"{column}_{agg}": multi_index_agg[(column, agg)]
        for column, agg in multi_index_agg.columns
    }).reset_index()

    # Only "count"/"sum" of an empty group are 0, for "mean"/"min"/"max"
    # a missing value means "no data", so leave it as NaN to avoid misleading the model.
    fill_zero_columns = [
        f"{column}_{agg}"
        for column, agg in multi_index_agg.columns
        if agg in ("count", "sum")
    ]
    agg_df[fill_zero_columns] = agg_df[fill_zero_columns].fillna(0)

    return agg_df

# The `agg_categorical_columns` function creates a new DataFrame with aggregated
# categorical columns of the given DataFrame.
def agg_categorical_columns(df, groupby_columns, exclude_columns=None):
    exclude_columns = exclude_columns or []
    categorical_columns = [
        c for c in df.select_dtypes(include="str").columns
        if c not in groupby_columns and c not in exclude_columns
    ]

    if not categorical_columns:
        return df[groupby_columns].drop_duplicates().reset_index(drop=True)
    
    one_hot_df = pd.get_dummies(df[groupby_columns + categorical_columns])
    multi_index_agg = one_hot_df.groupby(groupby_columns).agg(["sum", "mean"])
    multi_index_agg = multi_index_agg.rename(columns={"sum": "count", "mean": "count_norm"}, level=1)
    
    agg_df = pd.DataFrame({
        f"{column}_{agg}": multi_index_agg[(column, agg)]
        for column, agg in multi_index_agg.columns
    }).reset_index()
    
    return agg_df

# A method that aggregates both numeric and categorical features of a dataframe
def agg_columns(df, groupby_columns, exclude_columns=None, prefix=None):
    agg_numeric_df = agg_numeric_columns(df=df, groupby_columns=groupby_columns, exclude_columns=exclude_columns)
    agg_categorical_df = agg_categorical_columns(df=df, groupby_columns=groupby_columns, exclude_columns=exclude_columns)

    agg_df = pd.merge(agg_numeric_df, agg_categorical_df, on=groupby_columns)

    if prefix:
        agg_df = agg_df.rename(columns={
            column: f"{prefix}_{column}"
            for column in agg_df.columns
            if column not in groupby_columns
        })

    return agg_df

# Extract the columns that have no more than the threshold percentage of NaN values
def remove_nan_columns(df, threshold=0.3):
    valid_columns = (df.isna().sum() / len(df) <= threshold)
    return df.loc[:, valid_columns]
