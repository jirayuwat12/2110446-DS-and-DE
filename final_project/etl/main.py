import time
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers=['localhost:9092'])

while True:
    current_time_str = f'current time: {time.time()}'
    producer.send('test', current_time_str.encode())
    time.sleep(1)
