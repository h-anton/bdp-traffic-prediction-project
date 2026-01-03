#!/usr/bin/env python3
# /// script
# dependencies = [
#   "kafka-python-ng",
#   "pyarrow",
#   "pandas",
# ]
# ///

import argparse
import json
import time
import threading

import pyarrow.parquet as pq
from kafka import KafkaProducer


def parse_args():
    parser = argparse.ArgumentParser(description="Feed traffic data into Kafka.")
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers, e.g. localhost:9092",
    )
    parser.add_argument(
        "--speed",
        type=str,
        required=True,
        help="File with speed data.",
    )
    parser.add_argument(
        "--volume",
        type=str,
        required=True,
        help="File with volume data.",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=100,
        help="Sleep in milliseconds between messages (simulates real time).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to send.",
    )
    return parser.parse_args()


def message(row, node):
    return {
        "timestamp": row["timestamp"].isoformat(),
        "node_id": node,
        "value": float(row[node])
    }


def send(df, topic, args):
    print(f"Connecting to Kafka at {args.bootstrap_servers} ...")

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=str.encode
    )

    if not producer.bootstrap_connected():
        return

    print(f"Connected.\n")

    pushed = 0
    for idx, row in df.iterrows():
        for column in df.columns:
            if column not in ["timestamp"]:
                print(f"Send message to '{topic}' for key: {column} ...")
                future = producer.send(topic, key=column, value=message(row, column))
                result = future.get(timeout=60)
                print(result)
                pushed = pushed + 1
            if args.limit and pushed == args.limit:
                return
        producer.flush()

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    producer.close()


def main():
    args = parse_args()

    print(f"Reading data ...")
    speed = pq.read_table(args.speed).to_pandas()
    volume = pq.read_table(args.volume).to_pandas()

    print(speed.head())

    speed_thread = threading.Thread(target=send, args=(speed, "speed", args))
    volume_thread = threading.Thread(target=send, args=(volume, "volume", args))

    speed_thread.start()
    volume_thread.start()

    speed_thread.join()
    volume_thread.join()

    return


if __name__ == "__main__":
    main()
