import multiprocessing
import time

import boto3

# Initialize the S3 resource
s3 = boto3.resource("s3")


# Function to download a specific byte range
def download_range(bucket_name, key, byte_range):
    chunk_size = 500
    s3_object = s3.Object(bucket_name, key)
    return s3_object.get(Range=byte_range)["Body"].iter_chunks(chunk_size)


# Worker function to perform S3 requests
def worker(bucket_name, key, offset):
    byte_range = f"bytes={offset}-"
    try:
        start = time.time()
        data = download_range(bucket_name, key, byte_range)
        duration = time.time() - start
        print(f"Duration {duration*1e3}ms ")
    except Exception as e:
        print(f"Error in downloading range {byte_range}: {e}")


# Function to maximize S3 requests using multiprocessing
def max_requests(bucket_name, key, num_processes, chunk_size):
    object_size = s3.Object(bucket_name, key).content_length
    offsets = range(0, object_size, chunk_size)

    # Create a pool of workers
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(worker, [(bucket_name, key, offset) for offset in offsets])


if __name__ == "__main__":
    bucket_name = "test-aws-us-east-1"
    key = "hq_data_books3_2049_dd_alpha.gig.npy"

    # Number of processes to run in parallel
    num_processes = 1  # Adjust based on your machine's capacity and network speed

    # Size of each chunk to download in bytes
    chunk_size = 500

    max_requests(bucket_name, key, num_processes, chunk_size)
