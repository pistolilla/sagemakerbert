import os, json
import time
import boto3
timestamp = int(time.time())

import sagemaker
role = "AmazonSageMaker-ExecutionRole-20221226T140961"
AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
sess   = sagemaker.Session()

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)

RAW_INPUT_DATA_S3_URI = 's3://dlai-practical-data-science/data/raw/'

processing_instance_type = ParameterString(
    name="ProcessingInstanceType",
    default_value="ml.c5.2xlarge"
)

processing_instance_count = ParameterInteger(
    name="ProcessingInstanceCount",
    default_value=1
)

train_split_percentage = ParameterFloat(
    name="TrainSplitPercentage",
    default_value=0.90,
)

validation_split_percentage = ParameterFloat(
    name="ValidationSplitPercentage",
    default_value=0.05,
)

test_split_percentage = ParameterFloat(
    name="TestSplitPercentage",
    default_value=0.05,
)

balance_dataset = ParameterString(
    name="BalanceDataset",
    default_value="True",
)

max_seq_length = ParameterInteger(
    name="MaxSeqLength",
    default_value=128,
)

feature_store_offline_prefix = ParameterString(
    name="FeatureStoreOfflinePrefix",
    default_value="reviews-feature-store-" + str(timestamp),
)

feature_group_name = ParameterString(
    name="FeatureGroupName",
    default_value="reviews-feature-group-" + str(timestamp)
)

input_data = ParameterString(
    name="InputData",
    default_value=RAW_INPUT_DATA_S3_URI,
)

from sagemaker.sklearn.processing import SKLearnProcessor

processor = SKLearnProcessor(
    framework_version='0.23-1',
    role=role,
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    env={'AWS_DEFAULT_REGION': AWS_DEFAULT_REGION},
)

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

processing_inputs=[
    ProcessingInput(
        input_name='raw-input-data',
        source=input_data,
        destination='/opt/ml/processing/input/data/',
        s3_data_distribution_type='ShardedByS3Key'
    )
]

processing_outputs=[
    ProcessingOutput(output_name='sentiment-train',
        source='/opt/ml/processing/output/sentiment/train',
        s3_upload_mode='EndOfJob'),
    ProcessingOutput(output_name='sentiment-validation',
        source='/opt/ml/processing/output/sentiment/validation',
        s3_upload_mode='EndOfJob'),
    ProcessingOutput(output_name='sentiment-test',
        source='/opt/ml/processing/output/sentiment/test',
        s3_upload_mode='EndOfJob')
]

processing_step = ProcessingStep(
    name='Processing',
    code='src/prepare_data.py',
    processor=processor,
    inputs=processing_inputs,
    outputs=processing_outputs,
    job_arguments=[
        '--train-split-percentage', str(train_split_percentage.default_value),
        '--validation-split-percentage', str(validation_split_percentage.default_value),
        '--test-split-percentage', str(test_split_percentage.default_value),
        '--balance-dataset', str(balance_dataset.default_value),
        '--max-seq-length', str(max_seq_length.default_value),
        '--feature-store-offline-prefix', str(feature_store_offline_prefix.default_value),
        '--feature-group-name', str(feature_group_name.default_value)
    ]
)

print("Processing step (obj):")
print(processing_step)

print("List of the processing job properties:")
print(json.dumps(
    processing_step.properties.__dict__,
    indent=4, sort_keys=True, default=str
))

print("Processing step outputs (config):")
print(processing_step.arguments['ProcessingOutputConfig'])

from sagemaker.workflow.pipeline import Pipeline

pipeline_name = "miprimerpipeline08"

pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        input_data,
        processing_instance_count,
        processing_instance_type,
        max_seq_length,
        balance_dataset,
        train_split_percentage,
        validation_split_percentage,
        test_split_percentage,
        feature_store_offline_prefix,
        feature_group_name,
    ],
    steps=[processing_step],
    sagemaker_session=sess,
)

from pprint import pprint

definition = json.loads(pipeline.definition())

print("Pipeline definition:")
pprint(definition)

response = pipeline.create(role_arn="arn:aws:iam::058641753359:role/service-role/AmazonSageMaker-ExecutionRole-20221226T140961")

pipeline_arn = response["PipelineArn"]
print("Pipeline ARN:")
print(pipeline_arn)

print("Executing pipeline")
execution = pipeline.start(
    parameters=dict(
        InputData=RAW_INPUT_DATA_S3_URI,
        ProcessingInstanceCount=1,
        ProcessingInstanceType='ml.c5.2xlarge',
        MaxSeqLength=128,
        BalanceDataset='True',
        TrainSplitPercentage=0.9,
        ValidationSplitPercentage=0.05,
        TestSplitPercentage=0.05,
        FeatureStoreOfflinePrefix='reviews-feature-store-'+str(timestamp),
        FeatureGroupName='reviews-feature-group-'+str(timestamp),
    )
)
print("Pipeline ARN:")
print(execution.arn)

print("Describe execution:")
execution_run = execution.describe()
pprint(execution_run)

print("Poll execution status")
sm = boto3.Session().client(service_name='sagemaker', region_name=AWS_DEFAULT_REGION)

executions_response = sm.list_pipeline_executions(PipelineName=pipeline_name)['PipelineExecutionSummaries']
pipeline_execution_status = executions_response[0]['PipelineExecutionStatus']
print(pipeline_execution_status)

while pipeline_execution_status=='Executing':
    try:
        executions_response = sm.list_pipeline_executions(PipelineName=pipeline_name)['PipelineExecutionSummaries']
        pipeline_execution_status = executions_response[0]['PipelineExecutionStatus']
    except Exception as e:
        print('Please wait...')
        time.sleep(30)

pprint(executions_response)