import boto3
import os
import sagemaker
from sagemaker.debugger import Rule, rule_configs
from sagemaker.session import TrainingInput

region = boto3.Session().region_name

bucket = os.environ.get("s3_bucket")
prefix = os.environ.get("s3_prefix")

role = os.environ.get("sagemake_role")

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

print(container)

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

xgb_model.fit({"train": train_input, "validation": validation_input}, wait=True)
