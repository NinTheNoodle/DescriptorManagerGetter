"""
Authors:
  Rory Higgins: C12428088
  Shane Farrell: C12722625
"""
from pprint import pprint
from math import log2
from random import seed, shuffle
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


def to_list(dataframe):
    """
    Converts dataframe to a list of dictionaries, and adds the "id" column

    Unfortunately rather expensive, but necessary for DictVectorizer.
    DataFrame.to_dict is not used as it does not preserve order
    """
    rows = []
    for row_id, row in dataframe.iterrows():
        row = dict(row)
        row["id"] = row_id
        rows.append(row)

    return rows


def drop_column(rows, name):
    for row in rows:
        del row[name]


def get_column(rows, name):
    return [row[name] for row in rows]


def proces_columns(rows):
    drop_column(rows, "id")
    drop_column(rows, "feature")
    drop_column(rows, "duration")


def split(data):
    midway = len(data) // 2
    return data[:midway], data[midway:]


def load_data(fname, names):
    data = pd.read_csv(fname, index_col=0, names=names)
    return to_list(data)


def main():
    onehot_encoder = DictVectorizer(sparse=False)
    classifier = RandomForestClassifier(
        criterion="entropy"
    )

    training_rows = load_data("data/trainingset.txt", names)
    query_rows = load_data("data/queries.txt", names)

    seed(123)
    shuffle(training_rows)

    training_features = get_column(training_rows, "feature")
    query_ids = get_column(query_rows, "id")

    proces_columns(training_rows)
    proces_columns(query_rows)

    training_rows = onehot_encoder.fit_transform(training_rows)
    query_rows = onehot_encoder.fit_transform(query_rows)

    training_rows, validation_rows = split(training_rows)
    training_features, validation_features = split(training_features)

    classifier.fit(training_rows, training_features)
    accuracy = classifier.score(validation_rows, validation_features)

    # make target estimates
    result = classifier.predict(query_rows)
    result_data = pd.DataFrame(list(zip(query_ids, result)))
    result_data.to_csv("solutions/C12722625+C12428088.txt", header=False,
                       index=False, float_format="%.4f")

    ratio = len([row for row in result if row == "TypeA"]) / len(result)

    print("A/B ratio:", ratio)
    print("accuracy:", accuracy)


if __name__ == "__main__":
    main()
