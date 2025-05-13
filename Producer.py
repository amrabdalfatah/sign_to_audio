from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

order_event = {
    "user_id": 1001,
    "product_name": "Wireless Headphones",
    "status": "Completed"
}

producer.send('order_events', order_event)
producer.flush()
print("✅ Order event sent to Kafka.")
