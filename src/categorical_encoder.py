import pandas as pd 

# Catergorical encoder
def categorical_encoder(df):
    for column in df.select_dtypes(include=['object']).columns:

        df[column] = pd.Categorical(df[column]).codes

    return df