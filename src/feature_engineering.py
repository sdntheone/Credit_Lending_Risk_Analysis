from src.data_transformation import transformed_data
from src.data_ingestion import load_config
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


def encode_categorical(df, categorical_cols):
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    encoder = OrdinalEncoder()
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

    return df, categorical_cols


def encode_target(df, target_col):
    lb = LabelEncoder()
    df[target_col] = lb.fit_transform(df[target_col])
    return df


def encoded_data():
    config = load_config()

    target_col = config["features"]["target_column"]
    categorical_cols = config["features"]["categorical_cols"]

    # load transformed data
    df = transformed_data()

    # encode categorical features
    df, categorical_cols = encode_categorical(df, categorical_cols)

    # encode target
    df = encode_target(df, target_col)

    # create categorical indices for SMOTENC
    cat_indices = [df.columns.get_loc(col) for col in categorical_cols]

    return df, cat_indices


if __name__ == "__main__":
    df, cat_indices = encoded_data()
    print(df.shape)
    print("Categorical indices:", cat_indices)