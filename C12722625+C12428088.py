"""
Authors:
  Rory Higgins: C12428088
  Shane Farrell: C12722625
"""
from math import log2
from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

names = [
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


def main(fname="data/trainingset.txt", names=names):
    # import pdb; pdb.set_trace()

    # extract data
    data = pd.read_csv(fname, index_col=0, names=names)
    rows = list(data.to_dict(orient="index").values())

    # extract features
    features = [row["feature"] for row in rows]
    for row in rows:
        del row["feature"]

    # perform one-hot encoding
    vectorizer = DictVectorizer(sparse=False)
    rows = vectorizer.fit_transform(rows)

    # split into training and validation sets
    midway = len(rows) // 2
    training_rows = rows[:midway]
    training_features = features[:midway]
    validation_rows = rows[midway:]
    validation_features = features[midway:]

    classifier = RandomForestClassifier(
        criterion="entropy",
        max_depth=int(log2(len(rows))) // 2,
        n_jobs=8
    )
    classifier.fit(training_rows, training_features)

    accuracy = classifier.score(validation_rows, validation_features)
    print(accuracy)


if __name__ == "__main__":
    main()
