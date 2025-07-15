import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_otu_table(url: str):
    return pd.read_csv(url, index_col=0)

def create_labels(samples):
    return ["healthy" if i < len(samples)/2 else "sick" for i in range(len(samples))]

def prepare_data(otu_table, labels):
    otu_features = otu_table.T
    otu_features["label"] = labels
    X = otu_features.drop("label", axis=1)
    y = otu_features["label"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42), X.columns

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

def plot_feature_importance(model, feature_names, top_n=10):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:top_n]
    top_features = feature_names[indices]
    plt.figure(figsize=(10, 5))
    plt.bar(top_features, importances[indices])
    plt.xticks(rotation=45)
    plt.title("Top 10 Important OTUs")
    plt.tight_layout()
    plt.savefig("otu_importance.png")
    plt.close()
