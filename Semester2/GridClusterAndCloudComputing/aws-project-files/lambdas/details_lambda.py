import json
import boto3

dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')

TABLE_NAME = 'JournalEntries'
S3_BUCKET = 'journal-app-upload'

def lambda_handler(event, context):
    try:
        # Get entry_id from path
        entry_id = event["pathParameters"]["entry_id"]

        table = dynamodb.Table(TABLE_NAME)
        response = table.get_item(Key={"entry_id": entry_id})

        if "Item" not in response:
            return {
                "statusCode": 404,
                "body": json.dumps({"error": "Entry not found"})
            }

        item = response["Item"]
        result = {
            "entry_id": item["entry_id"],
            "timestamp": item["timestamp"],
            "text": item["text"]
        }

        if "file" in item:
            key = item["file"]["s3_key"]
            url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': S3_BUCKET, 'Key': key},
                ExpiresIn=3600
            )
            result["file"] = {
                "filename": item["file"]["filename"],
                "content_type": item["file"]["content_type"],
                "download_url": url
            }

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
