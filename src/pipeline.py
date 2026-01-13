import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import os
import joblib
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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

        # Input (X)
        X = df[['Age', 'Gender', 'BMI', 'Chol', 'TG', 'HDL', 'LDL', 'Cr', 'BUN', 'cluster']]
        y = df['Diagnosis']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

    def clustering(self):
        df = pd.read_csv(self.data_path)

        X = df[['Age', 'Gender', 'BMI', 'Chol', 'TG', 'HDL', 'LDL', 'Cr', 'BUN']]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # save scaler and cluster
        joblib.dump(scaler, "../models/scaler.pkl")
        joblib.dump(kmeans, "../models/kmeans.pkl")

        # add 'cluster' column before 'diagnosis' and save it in database
        df["cluster"] = clusters
        cols = df.columns.tolist()
        cols.insert(cols.index("Diagnosis"), cols.pop(cols.index("cluster")))
        df = df[cols]

        # split path name and proper save
        base, ext = os.path.splitext(self.data_path)
        df.to_csv(f"{base}_clusters{ext}", index=False)
        print(df.groupby("cluster").mean())

        # show clusters visualization
        plt.figure(figsize=(6, 4))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
        plt.title("PCA Visualization of Patient Clusters")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.show()

        # Testing proper value for k in clustering by elbow plot
        # ---
        # inertias = []
        # K_range = range(1, 11)
        #
        # for k in K_range:
        #     kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        #     kmeans.fit(X_scaled)
        #     inertias.append(kmeans.inertia_)
        #
        # # inertia dla K=1 (total variance)
        # inertia_1 = inertias[0]
        #
        # explained_variance = [
        #     1 - (i / inertia_1) for i in inertias
        # ]
        #
        # plt.plot(K_range, explained_variance, marker="o")
        # plt.xlabel("Number of clusters (K)")
        # plt.ylabel("Explained variance (R²)")
        # plt.title("Elbow method – variance explained")
        # plt.show()

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
clustering_trainer = ModelTrainer("../data/Diabetes_Classification_le.csv")
clustering_trainer.clustering()
xgb_trainer = ModelTrainer("../data/Diabetes_Classification_le_clusters.csv")
xgb_trainer.load_data()
# best_params = xgb_trainer.grid_search()
# print(best_params)
xgb_trainer.train(n_estimators=600,         # number of trees
            max_depth=4,                # max depth of trees
            learning_rate=0.01,         # learning rate aka eta
            colsample_bytree=0.7,       # is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
            subsample=0.7,              # Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the
                                        # training data prior to growing trees. and this will prevent overfitting.
                                        # Subsampling will occur once in every boosting iteration. range: (0,1]

            min_child_weight=3,         # Minimum sum of instance weight (hessian) needed in a child
            scale_pos_weight = 2      # Control the balance of positive and negative weights, typical value to consider: sum(negative instances) / sum(positive instances)
            )
xgb_trainer.evaluate()
xgb_trainer.save()

# Checking importance of columns
model = xgb_trainer.get_model()
plot_importance(model, importance_type="gain") # importance type gain/weight/cover
plt.show()


