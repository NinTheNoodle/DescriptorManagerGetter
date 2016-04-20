"""
Authors:
  Rory Higgins: C12428088
  Shane Farrell: C12722625

Artificial Intelligence 2 - Assignment 2

"Develop a classifier that uses data to
predict the outcome of a Bank marketing campaign."

"""
from random import seed, shuffle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

# Splits dataset, performing validation and computing an accuracy score
compute_accuracy = False

# random seed to use
random_seed = 123

# Descriptive feature names
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


def split(rows, frac=0.5):
    """
    Splits a given dataset into two

    frac: what fraction of the data should be in the first slice
    """
    pivot = int(len(rows) * frac)
    return rows[:pivot], rows[pivot:]


def to_list(dataframe):
    """
    Converts dataframe to a list of dictionaries, and adds the "id" column

    Unfortunately rather expensive, but necessary for DictVectorizer
    DataFrame.to_dict is not used as it does not preserve order
    """
    rows = []
    for row_id, row in dataframe.iterrows():
        row = dict(row)
        row["id"] = row_id
        rows.append(row)

    return rows


def load_data(fname, names):
    """
    Loads data from CSV file, returns it as a list of dictionaries
    """
    data = pd.read_csv(fname, index_col=0, names=names)
    return to_list(data)


def process_columns(rows):
    """
    Cleans the data to account for issues that could skew the results
    """
    for row in rows:
        # definitely doesn't belong in training data
        del row["id"]
        del row["feature"]

        # introducing a meaningful correlation from otherwise junk data
        row["contacted"] = (row["pdays"] == -1)

        # "-1" values would skew the results
        del row["pdays"]

        # all zero
        del row["duration"]

        # not really numeric (does not make sense to take the mean)
        del row["day"]


def main():
    onehot_encoder = DictVectorizer(sparse=False)
    classifier = RandomForestClassifier(
        random_state=random_seed,
        n_estimators=100,
        criterion="gini",
        n_jobs=8
    )

    training_rows = load_data("data/trainingset.txt", names)
    query_rows = load_data("data/queries.txt", names)

    seed(random_seed)
    shuffle(training_rows)

    training_features = [row["feature"] for row in training_rows]
    query_ids = [row["id"] for row in query_rows]

    process_columns(training_rows)
    process_columns(query_rows)

    training_rows = onehot_encoder.fit_transform(training_rows)
    query_rows = onehot_encoder.fit_transform(query_rows)

    if compute_accuracy:
        # split dataset into training and validation data sets
        training_rows, validation_rows = split(training_rows)
        training_features, validation_features = split(training_features)

    classifier.fit(training_rows, training_features)

    # make target estimates
    results = classifier.predict(query_rows)
    with open("solutions/C12722625+C12428088.txt", "w") as output:
        for query_id, result in zip(query_ids, results):
            output.write("{id},{result}\n".format(id=query_id, result=result))

    if compute_accuracy:
        accuracy = classifier.score(validation_rows, validation_features)
        print("accuracy:", accuracy)

        ratio = (len([row for row in training_features
                      if row == "TypeA"]) / len(training_features))
        print("Input A/B ratio:", ratio)

        ratio = len([row for row in results if row == "TypeA"]) / len(results)
        print("Result A/B ratio:", ratio)


if __name__ == "__main__":
    main()
