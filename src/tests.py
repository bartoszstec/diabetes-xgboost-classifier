import joblib
import numpy as np
import pandas as pd

# df = pd.read_csv("../data/Diabetes_Classification.csv")
#
# # create columns
# df["is_magic"] = (df[["Chol","TG","HDL","LDL","Cr","BUN"]] == 4.860753).any(axis=1)
# df["magic_count"] = (df[["Chol","TG","HDL","LDL","Cr","BUN"]] == 4.860753).sum(axis=1)
# print(df.groupby("is_magic")["Diagnosis"].value_counts(normalize=True).unstack())
# df.to_csv("../data/test.csv", index=False)

# load existing model and test particular prediction data
model = joblib.load("../models/bst_diabetes_classifier_clusters.pkl")
scaler = joblib.load("../models/scaler.pkl")
kmeans = joblib.load("../models/kmeans.pkl")
X = np.array([[37,1,34,5.42,2.66,1.08,2.87,75.5,4.61]])
# scaling
x_scaled = scaler.transform(X)

# defining a cluster for a patient
cluster = kmeans.predict(x_scaled)[0]

X_cluster = np.append(X, cluster.reshape(-1, 1), axis=1)
print(X_cluster)
prediction = model.predict(X_cluster)[0]
probability_0 = model.predict_proba(X_cluster)[0][0]
probability_1 = model.predict_proba(X_cluster)[0][1]

print(f"klasa: {prediction}")
print(f"prawdopodobieństwo nieposiadania cukrzycy: {probability_0}")
print(f"prawdopodobieństwo posiadania cukrzycy: {probability_1}")