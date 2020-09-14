"""
Script to train model.
"""
import logging
import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from bedrock_client.bedrock.analyzer.model_analyzer import ModelAnalyzer
from bedrock_client.bedrock.analyzer import ModelTypes
from bedrock_client.bedrock.api import BedrockApi

from constants import FEATURES, TARGET, CONFIG_FAI

TMP_BUCKET = "span-production-temp-data"  # DO NOT MODIFY

# Model must be saved in /artefact/
OUTPUT_MODEL_PATH = "/artefact/model.pkl"
# You can also save any other files and logs in /artefact/
# They can be retrieved together with the model by downloading from the UI

FEATURES_FILE = os.getenv("FEATURES_FILE")
N_ESTIMATORS = os.getenv("N_ESTIMATORS")
MAX_FEATURES = os.getenv("MAX_FEATURES")
BOOTSTRAP = os.getenv("BOOTSTRAP")
RANDOM_STATE = os.getenv("RANDOM_STATE")



def compute_log_metrics(clf, x_val, y_val):
    """Compute and log metrics."""
    y_prob = clf.predict_proba(x_val)[:, 1]
    
    # select best threshold
    fpr, tpr, thresholds = metrics.roc_curve(y_val, y_prob)
    best_threshold = thresholds[np.argmax(tpr-fpr)]
    
    y_pred = (y_prob > best_threshold).astype(int)

    acc = metrics.accuracy_score(y_val, y_pred)
    roc_auc = metrics.roc_auc_score(y_val, y_prob)
    avg_prc = metrics.average_precision_score(y_val, y_prob)
    print("Evaluation\n"
          f"  Accuracy          = {acc:.4f}\n"
          f"  ROC AUC           = {roc_auc:.4f}\n"
          f"  Average precision = {avg_prc:.4f}\n")
    print(metrics.classification_report(y_val, y_pred, digits=4))

    # Log metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("ROC AUC", roc_auc)
    bedrock.log_metric("Avg precision", avg_prc)
    bedrock.log_chart_data(y_val.astype(int).tolist(),
                           y_prob.flatten().tolist())

    # Calculate and upload xafai metrics
    analyzer = ModelAnalyzer(clf, "tree_model", model_type=ModelTypes.TREE).test_features(x_val)
    analyzer.test_labels(y_val).test_inference(y_pred)
    analyzer.analyze()


def main():
    """Entry point to perform training."""
    print("\nLoad data")
    df = pd.read_csv("data/features.csv")
    print("Data shape:", df.shape)

    X = df[FEATURES]
    Y = df[TARGET].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify = Y)
    
    print("\nTrain model")
    start = time.time()
    clf = RandomForestClassifier(n_estimators = int(N_ESTIMATORS), max_features = MAX_FEATURES, bootstrap = (BOOTSTRAP=='True'), random_state = int(RANDOM_STATE))
    clf.fit(X_train, Y_train)
    print(f"  Time taken = {time.time() - start:.0f} s")
    
    print("\nEvaluate")
    compute_log_metrics(clf, X_test, Y_test)
    
    print("\nSave model")
    with open(OUTPUT_MODEL_PATH, "wb") as model_file:
        pickle.dump(clf, model_file)


if __name__ == "__main__":
    main()