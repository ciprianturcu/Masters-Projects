import json
import boto3

dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')

TABLE_NAME = 'JournalEntries'
S3_BUCKET = 'journal-app-upload'

def lambda_handler(event, context):
    try:
        table = dynamodb.Table(TABLE_NAME)
        response = table.scan()
        items = response['Items']

        sorted_items = sorted(items, key=lambda x: x['timestamp'], reverse=True)

        result = []
        for item in sorted_items:
            entry = {
                "entry_id": item["entry_id"],
                "timestamp": item["timestamp"],
                "text": item["text"]
            }

            if "file" in item:
                filename = item["file"]["filename"].lower()
                if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.pdf')):
                    key = item["file"]["s3_key"]
                    url = s3.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': S3_BUCKET, 'Key': key},
                        ExpiresIn=3600
                    )
                    entry["file"] = {
                        "filename": filename,
                        "content_type": item["file"].get("content_type"),
                        "thumbnail_url": url
                    }

            result.append(entry)

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
