import numpy as np
import pandas as pd
import boto3
import os
from sklearn.model_selection import train_test_split
import sagemaker
from sagemaker.debugger import Rule, rule_configs
from sagemaker.session import TrainingInput
from sagemaker.serializers import CSVSerializer

# Get Features and Labels
features = ref("features")

label = ref("earns_more_than_50k")

label_vector = np.equal(label["above"].to_numpy(), 1.0)

# Split data into training, validation and testing
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

# Convert data to CSV and upload data to S3
train.to_csv("train.csv", index=False, header=False)
validation.to_csv("validation.csv", index=False, header=False)

bucket = os.environ.get("s3_bucket")
prefix = os.environ.get("s3_prefix")

boto3.Session().resource("s3").Bucket(bucket).Object(
    os.path.join(prefix, "data/train.csv")
).upload_file("train.csv")
boto3.Session().resource("s3").Bucket(bucket).Object(
    os.path.join(prefix, "data/validation.csv")
).upload_file("validation.csv")

print("Finished preparing data")

# SageMaker training
role = os.environ.get("sagemaker_role")
region = sagemaker.Session().boto_region_name

s3_output_location = "s3://{}/{}/{}".format(bucket, prefix, "xgboost_model")

container = sagemaker.image_uris.retrieve("xgboost", region, "1.2-1")

xgb_model = sagemaker.estimator.Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    volume_size=5,
    output_path=s3_output_location,
    sagemaker_session=sagemaker.Session(),
    rules=[Rule.sagemaker(rule_configs.create_xgboost_report())],
)

xgb_model.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.7,
    objective="binary:logistic",
    num_round=1000,
)

train_input = TrainingInput(
    "s3://{}/{}/{}".format(bucket, prefix, "data/train.csv"), content_type="csv"
)

validation_input = TrainingInput(
    "s3://{}/{}/{}".format(bucket, prefix, "data/validation.csv"), content_type="csv"
)

print("Starting training")

xgb_model.fit({"train": train_input, "validation": validation_input}, wait=True)

print("Training complete")

# Model deployment

xgb_predictor = xgb_model.deploy(
    initial_instance_count=1, instance_type="ml.t2.medium", serializer=CSVSerializer()
)

print(xgb_predictor.endpoint_name)
