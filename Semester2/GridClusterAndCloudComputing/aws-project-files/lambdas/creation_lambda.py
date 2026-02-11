import json
import base64
import uuid
import boto3
from datetime import datetime

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

S3_BUCKET = 'journal-app-upload'
DYNAMODB_TABLE = 'JournalEntries'

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        journal_text = body.get('text')

        if not journal_text:
            return response(400, "Missing 'text'")

        file_data = body.get('file_data')  # base64
        file_name = body.get('file_name')
        content_type = body.get('content_type')

        entry_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        s3_key = None
        if file_data and file_name:
            decoded_file = base64.b64decode(file_data)
            s3_key = f"uploads/{entry_id}/{file_name}"

            s3.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=decoded_file,
                ContentType=content_type or 'application/octet-stream'
            )

        # Save unified entry
        item = {
            "entry_id": entry_id,
            "text": journal_text,
            "timestamp": timestamp,
        }

        if s3_key:
            item["file"] = {
                "s3_key": s3_key,
                "filename": file_name,
                "content_type": content_type
            }

        table = dynamodb.Table(DYNAMODB_TABLE)
        table.put_item(Item=item)

        return response(200, {"message": "Saved entry", "entry_id": entry_id})

    except Exception as e:
        return response(500, f"Server error: {str(e)}")

def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body)
    }