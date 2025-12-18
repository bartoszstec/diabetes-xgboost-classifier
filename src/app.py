import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
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
        },
        "classes": {
        },
        "training_params": {
            "objective": report['params']['objective'],
            "eval_metric": report['params']['eval_metric'],
            "n_estimators": report['params']['n_estimators'],
            "max_depth": report['params']['max_depth'],
            "min_child_weight": report['params']['min_child_weight'],
            "learning_rate": report['params']['learning_rate'],
            "colsample_bytree": report['params']['colsample_bytree'],
            "subsample": report['params']['subsample'],
            "scale_pos_weight": report['params']['scale_pos_weight']
        },
    }

    class_metrics = {}
    for key in report:
        if key not in ['accuracy', 'macro avg', 'weighted avg', 'params'] and isinstance(report[key], dict):
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
            objective='binary:logistic',
            eval_metric='auc',
            enable_categorical=True
        )
        default_params.update(model_params)

        self.model = XGBClassifier(**default_params)
        self.model.fit(self.X_train, self.y_train)

    def grid_search(self):
        """Hyperparameter tuning using GridSearchCV for XGBoost."""

        param_grid = {
            "n_estimators": [100, 200, 400, 600, 800],
            "max_depth": [3, 4, 6],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
            "min_child_weight": [1, 3, 5]
        }

        base_params = dict(
            objective="binary:logistic",
            eval_metric="auc",
            enable_categorical=True,
        )

        model = XGBClassifier(**base_params)

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="f1_macro",  # options: accuracy/f1_macro/roc_auc
            cv=3,
            n_jobs=-1,
            verbose=2
        )

        grid.fit(self.X_train, self.y_train)
        self.model = grid.best_estimator_

        print("===== BEST PARAMS =====")
        print(grid.best_params_)

        return grid.best_params_

    def evaluate(self):
        preds = self.model.predict(self.X_test)
        self.report_dict = classification_report(self.y_test, preds, output_dict=True)
        #self.report_dict["params"] = self.model.get_xgb_params()   #xgb
        self.report_dict["params"] = self.model.get_params()        #scikit-learn
        report_string = classification_report(self.y_test, preds)
        matrix = confusion_matrix(self.y_test, preds)
        print(matrix)
        print(report_string)

    def save(self, model_name="bst_diabetes_classifier"):
        file_name = save_model_unique(self.model, model_name)
        save_training_stats(self.report_dict, file_name)
        return file_name

    def get_model(self):
        return self.model


# Tests
# ---
trainer = ModelTrainer("../data/Diabetes_Classification.csv")
trainer.load_data()
# best_params = trainer.grid_search()
# print(best_params)
trainer.train(n_estimators=400,         # number of trees
            max_depth=4,                # max depth of trees
            learning_rate=0.01,         # learning rate aka eta
            colsample_bytree=1.0,       # is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
            subsample=1.0,              # Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the
                                        # training data prior to growing trees. and this will prevent overfitting.
                                        # Subsampling will occur once in every boosting iteration. range: (0,1]

            min_child_weight=3,         # Minimum sum of instance weight (hessian) needed in a child
            scale_pos_weight = 1.6      # Control the balance of positive and negative weights, typical value to consider: sum(negative instances) / sum(positive instances)
            )
trainer.evaluate()
trainer.save()

# Checking importance of columns
model = trainer.get_model()
plot_importance(model, importance_type="gain")
plt.show()


