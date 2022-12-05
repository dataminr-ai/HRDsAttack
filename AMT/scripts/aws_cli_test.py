import boto3

if __name__ == '__main__':

    region_name = "us-east-1"
    endpoint_url = "https://mturk-requester.us-east-1.amazonaws.com"

    client = boto3.client(
        "mturk",
        endpoint_url=endpoint_url,
        region_name=region_name,
    )

    print(client.get_account_balance()["AvailableBalance"])