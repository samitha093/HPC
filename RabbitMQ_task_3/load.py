import pika
import numpy as np
from gen import genarateData
import csv
import os

DataFile = "data.csv"
DataSetSize = 31*40

# This code is checking if a file named "data.csv" exists. If the file does not exist, it generates
# data using the `genarateData()` function with a specified data set size. After that, it establishes
# a connection to a RabbitMQ server using the provided credentials and connection parameters. It then
# declares a queue named "K_queue" on the RabbitMQ server. Next, it reads data from the CSV file and
# sends each row as a message to the "K_queue" queue. Finally, it closes the connection to the
# RabbitMQ server.
if not os.path.exists(DataFile):
    print(f"The file '{DataFile}' does not exist.")
    genarateData(DataSetSize)
    print(f"The file '{DataFile}' genarated.")

# RabbitMQ server connection parameters
credentials = pika.PlainCredentials('admin', 'admin')
connection_params = pika.ConnectionParameters('172.28.128.1', credentials=credentials)
print(f"Connecting to RabbitMQ server: {connection_params.host}")

try:
    # Establish a connection to RabbitMQ server
    connection = pika.BlockingConnection(connection_params)
    channel = connection.channel()
    print(f"Connected to RabbitMQ server: {connection_params.host}")

    # Declare a queue
    channel.queue_declare(queue='K_queue')
    print(f"Queue 'K_queue' declared")

    # Read data from CSV file and send each row as a message
    with open(DataFile, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # Send the row as a message to the queue
            message = ','.join(row)
            channel.basic_publish(exchange='', routing_key='K_queue', body=message)
            print(f"Sent: {message}")

    # Close the connection
    connection.close()
    print(f"Connection closed")

except Exception as e:
    print(f"An error occurred: {e}")

# python3 load.py