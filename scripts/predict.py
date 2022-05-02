import boto3
import sagemaker
import os
from io import BytesIO
import numpy as np
import pandas as pd

bucket = os.environ.get("s3_bucket")
prefix = os.environ.get("s3_prefix")

batch_df = ref(context.current_model.name)
batch_df = batch_df.drop("Above50k", 1)
batch_df.to_csv("batch.csv", index=False, header=False)

print("Uploading batch data")
boto3.Session().resource("s3").Bucket(bucket).Object(
    os.path.join(prefix, "batch/batch.csv")
).upload_file("batch.csv")


# Batch input
batch_input = "s3://{}/{}/batch".format(bucket, prefix)

# Batch transform output
batch_output = "s3://{}/{}/batch-prediction".format(bucket, prefix)

print("Setting up the Sagemaker transformer")
sagemaker_models = ref("sagemaker_models")

model_name = sagemaker_models.sort_values(by="created_at", ascending=False).iloc[0][
    "job_name"
]

sagemaker_model = sagemaker.estimator.Estimator.attach(model_name)

transformer = sagemaker_model.transformer(
    instance_count=1, instance_type="ml.m4.xlarge", output_path=batch_output
)

print("Start prediction")

transformer.transform(
    data=batch_input, data_type="S3Prefix", content_type="text/csv", split_type="Line"
)

transformer.wait()

print("Prediction complete")

prediction_obj = (
    boto3.Session()
    .resource("s3")
    .Bucket(bucket)
    .Object(os.path.join(prefix, "batch-prediction/batch.csv.out"))
)

print("Downloading predictions")
with BytesIO(prediction_obj.get()["Body"].read()) as prediction_raw:
    predictions = np.loadtxt(prediction_raw, dtype="float")
    output_df = pd.concat(
        [
            batch_df,
            pd.Series(predictions, index=batch_df.index, name="Above50k", dtype=float),
        ],
        axis=1,
    )

    print("Writing predictions to the Data Warehouse")
    write_to_model(output_df, mode="overwrite")
