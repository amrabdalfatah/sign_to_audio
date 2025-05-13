from kafka import KafkaConsumer
import json
from pymongo import MongoClient

# Kafka consumer setup
consumer = KafkaConsumer(
    'order-events',
    bootstrap_servers='kafka:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

# Connect to MongoDB
client = MongoClient("mongodb://mongodb:27017/")
db = client["products"]
collection = db["products"]

print("🔁 Waiting for events...")

for message in consumer:
    order = message.value
    print("🟢 Received event:", order)

    # Update product popularity
    collection.update_one(
        {"product_id": order["product_id"]},
        {"$inc": {"popularity": 1}},
        upsert=True
    )
