import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import os
import joblib
import json
from datetime import datetime


# HELPER FUNCTIONS
# ---
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


# Pipeline class
# ---
class ModelTrainer():
    def __init__(self, data_path, seed=42):
        self.data_path = data_path
        self.seed = seed
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.report_dict = None

    def load_data(self):
        df = pd.read_csv(self.data_path)

        # Columns verification
        required_columns = {'Id', 'Age', 'Gender', 'BMI', 'Chol', 'TG', 'HDL', 'LDL', 'Cr', 'BUN', 'Diagnosis'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Plik CSV musi zawierać kolumny: {required_columns}")

        # Set 'Gender' column as categorical
        df['Gender'] = df['Gender'].astype("category")

        # Input (X)
        X = df[['Age', 'Gender', 'BMI', 'Chol', 'TG', 'HDL', 'LDL', 'Cr', 'BUN']]
        y = df['Diagnosis']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

    def train(self, **model_params):
        default_params = dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            enable_categorical=True,
            objective='binary:logistic',
            eval_metric='auc'
        )
        default_params.update(model_params)

        self.model = XGBClassifier(**default_params)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        preds = self.model.predict(self.X_test)
        self.report_dict = classification_report(self.y_test, preds, output_dict=True)
        report_string = classification_report(self.y_test, preds)
        return report_string

    def save(self, model_name="bst_diabetes_classifier"):
        file_name = save_model_unique(self.model, model_name)
        save_training_stats(self.report_dict, file_name)
        return file_name


# Tests
# ---
trainer = ModelTrainer("../data/Diabetes_Classification.csv")
trainer.load_data()
trainer.train()
report = trainer.evaluate()
print(report)
trainer.save()


