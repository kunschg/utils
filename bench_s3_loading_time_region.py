import time

import boto3
import numpy as np
import torch

chunk_size = 2049 * 32  #  seq_len x nb_bits


def get_data_iter(bucket_name, object_key):
    s3_resource = boto3.resource("s3")
    s3_object = s3_resource.Object(bucket_name=bucket_name, key=object_key)
    range_header = f"bytes={chunk_size}-"
    data_iter = s3_object.get(Range=range_header)["Body"].iter_chunks(chunk_size)
    for cloud_bytes in data_iter:
        yield torch.as_tensor(
            np.array(np.frombuffer(cloud_bytes, np.int32)), dtype=torch.long
        )


# Function to time the download of an object from S3
def time_s3_download(bucket_name, object_key):
    data_iter = get_data_iter(bucket_name, object_key)
    start_time = time.time()
    data = next(data_iter)
    end_time = time.time()

    return end_time - start_time


# Benchmark function for multiple regions
def benchmark_s3_regions(object_key, regions, samples):
    timings = {}

    for _ in range(samples):
        for region in regions:
            bucket_name = f"test-aws-{region}"
            download_time = time_s3_download(bucket_name, object_key)
            if region in timings.keys():
                timings[region].append(download_time)
            else:
                timings[region] = [download_time]

    for region in timings:
        average = sum(timings[region]) / len(timings[region])
        std = np.sqrt(
            sum([(elem - average) ** 2 for elem in timings[region]])
            / len(timings[region])
        )
        print(f"Region {region} time {average} std {std}")

    return timings


# Example usage
object_key = "hq_data_books3_2049_dd_alpha.gig.npy"
regions = ["us-east-1", "us-east-2", "ap-northeast-1"]
samples = 50

timings = benchmark_s3_regions(object_key, regions, samples)
