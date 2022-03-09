
sklearn-flyermlops-template
=================================


user: Steven
user_email: steven.forrester@delta.com


# test sam deploy
sam build lassofunction --template inference/deploy_template.yaml #Build only the called function

export ACCESS_POINT_ARN=arn:aws:elasticfilesystem:us-east-1:080196648158:access-point/fsap-0e5bc07c249aab2a5
export DEPLOY_LAMBDA_ARN=arn:aws:iam::080196648158:role/delegate-admin-deploy-lambda-fmops-template-role
export PROJECT_NAME=fmops-template
export BUCKET_NAME=dl-use1-opsdata-080196648158-flyermlops
sam deploy \
    --stack-name fmops-template-lasso-sam \
    --capabilities CAPABILITY_NAMED_IAM \
    --no-confirm-changeset \
    --parameter-overrides ModelName=lasso Version=318 Stage=stage EfsAccessPointArn=$ACCESS_POINT_ARN DeployLambdaRoleArn=$DEPLOY_LAMBDA_ARN ProjectName=$PROJECT_NAME \
    --s3-bucket $BUCKET_NAME \
    --s3-prefix project/fmops-template/sam/lasso \
    --no-fail-on-empty-changeset