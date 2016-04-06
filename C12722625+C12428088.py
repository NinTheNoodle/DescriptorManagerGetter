"""
Authors:
  Rory Higgins: C12428088
  Shane Farrell: C12722625
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
from pprint import pprint

headers = [
    "id",
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "feature"
]


def get_training_data(data):
    rows = data.to_dict(orient="index").values()
    for row in rows:
        del row["feature"]

    vectorizer = DictVectorizer(sparse=False)
    training_data = vectorizer.fit_transform(rows)

    return training_data, vectorizer


def main():
    data = pd.read_csv("data/trainingset.txt", index_col=0, names=headers)

    training_features = data["feature"]
    training_data, vectorizer = get_training_data(data)

    classifier = DecisionTreeClassifier(max_depth=3, criterion="entropy")
    classifier.fit(training_data, training_features)

    export_graphviz(classifier.tree_,
                    out_file='tree_d1.dot',
                    feature_names=training_features)

if __name__ == "__main__":
    main()
