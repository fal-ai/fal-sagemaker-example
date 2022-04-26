import numpy as np
import pandas as pd
import boto3
import os
from sklearn.model_selection import train_test_split

features = ref(context.current_model.name)

label = ref("earns_more_than_50k")
label_vector = np.equal(label["above"].to_numpy(), 1.0)

X_train, X_test, y_train, y_test = train_test_split(
    features, label_vector, test_size=0.2, random_state=1
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1
)

train = pd.concat(
    [pd.Series(y_train, index=X_train.index, name="Income>50K", dtype=int), X_train],
    axis=1,
)

validation = pd.concat(
    [pd.Series(y_val, index=X_val.index, name="Income>50K", dtype=int), X_val], axis=1
)

test = pd.concat(
    [pd.Series(y_test, index=X_test.index, name="Income>50K", dtype=int), X_test],
    axis=1,
)

train.to_csv("train.csv", index=False, header=False)
validation.to_csv("validation.csv", index=False, header=False)

bucket = "mederbucket"
prefix = "fal-sagemaker-income-prediction"

boto3.Session().resource("s3").Bucket(bucket).Object(
    os.path.join(prefix, "data/train.csv")
).upload_file("train.csv")
boto3.Session().resource("s3").Bucket(bucket).Object(
    os.path.join(prefix, "data/validation.csv")
).upload_file("validation.csv")
