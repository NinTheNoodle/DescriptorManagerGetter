"""
Authors:
  Rory Higgins: C12428088
  Shane Farrell: C12722625
"""
from pprint import pprint
from math import log2
import numpy as np
import pandas as pd
from sklearn.tree import export_graphviz
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


def dataframe_to_list(dataframe):
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


def drop_column(rows, column_name):
    """
    Extracts the values from a particular column and removes it from rows

    The type of rows is expected to be a list of dictionaries
    """
    values = []
    for row in rows:
        values.append(row[column_name])
        del row[column_name]

    return values


def main(data_fname="data/trainingset.txt",
         query_fname="data/queries.txt",
         names=names):
    # import pdb; pdb.set_trace()

    # extract data
    data = pd.read_csv(data_fname, index_col=0, names=names)
    rows = dataframe_to_list(data)
    drop_column(rows, "id")
    features = drop_column(rows, "feature")

    # perform one-hot encoding
    vectorizer = DictVectorizer(sparse=False)
    rows = vectorizer.fit_transform(rows)

    # split into training and validation sets
    midway = len(rows) // 2
    training_rows = rows[:midway]
    training_features = features[:midway]
    validation_rows = rows[midway:]
    validation_features = features[midway:]

    # generate random forest for data classification
    classifier = RandomForestClassifier(
        criterion="entropy",
        max_depth=int(log2(len(rows))) // 2,
        n_jobs=8
    )
    classifier.fit(training_rows, training_features)

    accuracy = classifier.score(validation_rows, validation_features)
    print("accuracy", accuracy)

    # extract queries
    query_data = pd.read_csv(query_fname, index_col=0, names=names)
    query_ids = list(query_data.index)
    queries = list(query_data.to_dict(orient="index").values())

if __name__ == "__main__":
    main()
