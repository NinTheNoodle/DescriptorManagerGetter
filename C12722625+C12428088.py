"""
Authors:
  Rory Higgins: C12428088
  Shane Farrell: C12722625
"""
from pprint import pprint
from math import log2
from random import shuffle
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


def get_column_processes():
    return {
        "duration": drop_column
    }


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
    process_columns = get_column_processes()

    vectorizer = DictVectorizer(sparse=False)

    # extract data
    data = pd.read_csv(data_fname, index_col=0, names=names)
    rows = to_list(data)
    shuffle(rows)

    drop_column(rows, "id")

    for column, action in process_columns.items():
        action(rows, column)

    features = drop_column(rows, "feature")

    rows = vectorizer.fit_transform(rows)

    # split into training and validation sets
    midway = len(rows) // 2
    training_rows = rows[:midway]
    training_features = features[:midway]
    validation_rows = rows[midway:]
    validation_features = features[midway:]

    # generate random forest for data classification
    classifier = RandomForestClassifier(
        criterion="entropy"
    )
    classifier.fit(training_rows, training_features)

    accuracy = classifier.score(validation_rows, validation_features)

    # extract queries
    query_data = pd.read_csv(query_fname, index_col=0, names=names)
    query_rows = to_list(query_data)
    query_ids = drop_column(query_rows, "id")

    for column, action in process_columns.items():
        action(query_rows, column)

    drop_column(query_rows, "feature")
    query_rows = vectorizer.fit_transform(query_rows)

    result = classifier.predict(query_rows)

    ratio = len([row for row in result if row == "TypeA"]) / len(result)
    print("A/B ratio:", ratio)
    print("accuracy:", accuracy)

if __name__ == "__main__":
    main()
