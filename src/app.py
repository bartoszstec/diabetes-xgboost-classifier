import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import os
import joblib
import json
from datetime import datetime



def save_model_unique(model, model_name="bst_diabetes_classifier"):
    """
    Saves model with unique name.

    Args:
        model: model object
        model_name: name of model

    Returns:
        str: name of file which contains model
    """

    # Base name of the file
    model_file = os.path.join(f"{model_name}.pkl")
    # Base path
    base_path = os.path.join("..", "models")
    base_filename = os.path.join(base_path, model_file)

    # Save if name available
    if not os.path.exists(base_filename):
        joblib.dump(model, base_filename)
        print(f"Model zapisany jako: {base_filename}")
        return model_file

    # Find first available name and save
    counter = 1
    new_filename = os.path.join(base_path, f"{model_name}_{counter}.pkl")

    while os.path.exists(new_filename):
        counter += 1
        new_filename = os.path.join(base_path, f"{model_name}_{counter}.pkl")
    model_file = os.path.join(f"{model_name}_{counter}.pkl")
    joblib.dump(model, new_filename)
    print(f"Model zapisano: {new_filename}")
    return model_file



def save_training_stats(report, file_name):
    """
    Saves model-trained statistics to JSON.

    Args:
        report: classification_report stored as dict
        file_name: name of file that contains model
    """

    filename = os.path.join("..", "models", "effectiveness_log.json")
    data = []

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []

    # Prepare stats
    stats = {
        "model": file_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "accuracy": float(f"{report['accuracy']:.4f}"),
        "metrics": {
            "weighted_avg": {
                "precision": float(f"{report['weighted avg']['precision']:.4f}"),
                "recall": float(f"{report['weighted avg']['recall']:.4f}"),
                "f1_score": float(f"{report['weighted avg']['f1-score']:.4f}")
            },
            "macro_avg": {
                "precision": float(f"{report['macro avg']['precision']:.4f}"),
                "recall": float(f"{report['macro avg']['recall']:.4f}"),
                "f1_score": float(f"{report['macro avg']['f1-score']:.4f}")
            }
        }
    }

    class_metrics = {}
    for key in report:
        if key not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(report[key], dict):
            class_metrics[key] = {
                "precision": float(f"{report[key].get('precision', 0):.4f}"),
                "recall": float(f"{report[key].get('recall', 0):.4f}"),
                "f1_score": float(f"{report[key].get('f1-score', 0):.4f}"),
                "support": int(report[key].get('support', 0))
            }
    stats["classes"] = class_metrics

    # Add stats
    data.append(stats)

    # Zapisz
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Statystyki dla {file_name} zapisano w pliku {filename}:")


# Wczytanie danych z pliku CSV
file_path = "../data/Diabetes_Classification.csv"   # Data file path
df = pd.read_csv(file_path)                         # Dataframe created and asigned
df_first_6 = df.head(6)
#print(df_first_6)
#print(df.describe())

# Columns verification
required_columns = {'Id', 'Age', 'Gender', 'BMI', 'Chol', 'TG', 'HDL', 'LDL', 'Cr', 'BUN', 'Diagnosis'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Plik CSV musi zawierać kolumny: {required_columns}")

# Set 'Gender' column as categorical
df['Gender'] = df['Gender'].astype('category')
# Input (X)
X = df[['Age', 'Gender', 'BMI', 'Chol', 'TG', 'HDL', 'LDL', 'Cr', 'BUN']]

# Prediction target (y)
y_class = df['Diagnosis']

SEED = 42
# Dataset split -> test/train
X_train, X_test,y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=SEED
)

# print(X_test)
# print(X_train)
# print(y_test)
# print(y_train)

# create model instance
bst = XGBClassifier(n_estimators=2, max_depth=2, enable_categorical=True, learning_rate=1, objective='binary:logistic', eval_metric='auc')
# fit model
bst.fit(X_train, y_train)
# make predictions
preds = bst.predict(X_test)
print(preds)

# # Ocena modelu
print("Macierz konfuzji:")
confusion_matrix = confusion_matrix(y_test, preds)
print(confusion_matrix)

print("\nRaport klasyfikacji:")
report_string = classification_report(y_test, preds)
report_dict = classification_report(y_test, preds, output_dict=True)
print(report_string)


# saving model and stats from report
model_file = save_model_unique(bst)
save_training_stats(report_dict, model_file)
#save_training_stats(report_dict, save_model_unique(bst))




