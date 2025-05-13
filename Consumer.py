from kafka import KafkaConsumer
import json
from pymongo import MongoClient

consumer = KafkaConsumer(
    'order_events',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["hybrid_db"]

for message in consumer:
    event = message.value
    product_name = event['product_name']

    db.products.update_one(
        {"name": product_name},
        {"$inc": {"popularity": 1}}
    )

    print(f"✅ Popularity updated for: {product_name}")
