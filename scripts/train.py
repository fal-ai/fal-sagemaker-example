"""Run prediction model training job and store model data in a dbt."""

import pandas as pd
import boto3
import os
from sklearn.model_selection import train_test_split
import sagemaker
from sagemaker.debugger import Rule, rule_configs
from sagemaker.session import TrainingInput
import time

# Get Features and Labels
training_sample = ref("training_sample")

labels_vector = training_sample["Income>50K"].to_numpy()
features = training_sample.drop("Income>50K", 1)

# Split data into training, validation
X_train, X_val, y_train, y_val = train_test_split(
    features, labels_vector, test_size=0.25, random_state=1
)

train = pd.concat(
    [pd.Series(y_train, index=X_train.index, name="Income>50K", dtype=int), X_train],
    axis=1,
)

validation = pd.concat(
    [pd.Series(y_val, index=X_val.index, name="Income>50K", dtype=int), X_val], axis=1
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

sagemaker_model = sagemaker.estimator.Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    volume_size=5,
    output_path=s3_output_location,
    sagemaker_session=sagemaker.Session(),
    rules=[Rule.sagemaker(rule_configs.create_xgboost_report())],
)

sagemaker_model.set_hyperparameters(
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

sagemaker_model.fit({"train": train_input, "validation": validation_input}, wait=True)

data = {
    "dbt_model": [context.current_model.name],
    "created_at": [time.time()],
    "job_name": [sagemaker_model.latest_training_job.name],
}

model_df = pd.DataFrame.from_dict(data)

write_to_model(model_df, mode="append")

print(f"{context.current_model.name} has been updated.")
