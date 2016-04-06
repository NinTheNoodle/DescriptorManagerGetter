"""
Authors:
  Rory Higgins: C12428088
  Shane Farrell: C12722625
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
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

def main():
    #with open("./data/trainingset.txt", "rb") as fl:
    #    raw_data = fl.read()
    #data = np.loadtxt(raw_data, delimiter=",")
    data = pd.read_csv("data/trainingset.txt", index_col=0, names=headers)
    
    training_data = data.to_dict(orient="index")#[x: data[x] for x in training_headers]
    training_feature = data["feature"]
    
    pprint(list(training_data.values())[:5])
    #classifier = DecisionTreeClassifier(max_depth=3, criterion="entropy")
    #classifier.fit(data.data, data.target)
    
    #export_graphviz(classifier.tree_, out_file='tree_d1.dot', feature_names=iris.feature_names)


if __name__ == "__main__":
    main()
