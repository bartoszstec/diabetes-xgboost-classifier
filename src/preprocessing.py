import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# data paths
source_path = "../data/Diabetes_Classification.csv"
normalized_path = "../data/Diabetes_Classification_normalized.csv"
sample_path = "../data/Diabetes_Classification_sample.csv"
le_path = "../data/Diabetes_Classification_label_encoding.csv"
ohe_path = "../data/Diabetes_Classification_one_hot_encoding.csv"

# Read main data -> normalize and save it
df = pd.read_csv(source_path)
df["Gender"] = df["Gender"].str.upper().str.strip()
df.to_csv(normalized_path, index=False)

# Save sample data
df_first_100 = df.head(101)
df_first_100.to_csv(sample_path, index=False)

# Label encoding for 'Gender' column and save it to file
df = pd.read_csv(normalized_path)
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df.to_csv(le_path, index=False)

# One-hot encoding for 'Gender' column and save it to file
df = pd.read_csv(normalized_path)
categorical_cols = ['Gender']
ohe = OneHotEncoder(sparse_output=False)
ohe_array = ohe.fit_transform(df[categorical_cols])

#print("OHE feature names:", ohe.get_feature_names_out(categorical_cols))
ohe_df = pd.DataFrame(
    ohe_array, columns=ohe.get_feature_names_out(categorical_cols))     # creating DataFrame object with encoded columns
df_ohe = pd.concat(
    [
        df.iloc[:, :3],     # adding encoded columns after 'Gender' column
        ohe_df,
        df.iloc[:, 3:]
    ],
    axis=1
)
df_ohe = df_ohe.drop(columns=categorical_cols)      # removing original 'Gender' column
df_ohe.to_csv(ohe_path, index=False)
