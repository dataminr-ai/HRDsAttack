import boto3

if __name__ == '__main__':

    region_name = "us-east-1"
    endpoint_url = "https://mturk-requester.us-east-1.amazonaws.com"
    subject_line = "{PUT YOUR SUBJECT HERE}"
    message_text = """{PUT YOUR MESSAGE HERE}"""
    worker_ids = []

    client = boto3.client(
        "mturk",
        endpoint_url=endpoint_url,
        region_name=region_name,
    )

    print(client.get_account_balance()["AvailableBalance"])

    response = client.notify_workers(
        Subject=subject_line,
        MessageText=message_text,
        WorkerIds=worker_ids,
    )

    print(f"Subject Line: {subject_line}")
    print(f"Message Text: {message_text}")
    print(f"Worker IDs: {', '.join(worker_ids)}")
    print(response)
