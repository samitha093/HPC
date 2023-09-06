import sys
import pika
import signal
import numpy as np

# The RabbitMQConsumer class is a Python class that connects to a RabbitMQ server, consumes messages
# from a specified queue, and stores the received messages as NumPy arrays.
class RabbitMQConsumer:
    def __init__(self, queue_name, username, password, host, batchSetSize):
        self.credentials = pika.PlainCredentials(username, password)
        self.connection_params = pika.ConnectionParameters(host, credentials=self.credentials)
        self.queue_name = queue_name
        self.messages_received = []
        self.connection = None
        self.channel = None
        self.length = batchSetSize
        self._rank = 0

    def connect(self):
        """
        The function establishes a connection to a RabbitMQ server and sets up a consumer to receive
        messages from a specified queue.
        """
        try:
            # Establish a connection to RabbitMQ server
            self.connection = pika.BlockingConnection(self.connection_params)
            self.channel = self.connection.channel()

            # Declare the queue
            self.channel.queue_declare(queue=self.queue_name)

            # Set up a consumer to receive messages
            self.channel.basic_consume(queue=self.queue_name, on_message_callback=self.callback)

        except Exception as e:
            print(f"An error occurred while connecting to RabbitMQ: {e}")

    def callback(self, ch, method, properties, body):
        """
        The callback function receives a message, converts it to a NumPy array, stores it, and stops
        consuming messages if the desired number has been received.
        
        Args:
          ch: The parameter "ch" is the channel object, which is used to communicate with the RabbitMQ
        server. It provides methods for sending and receiving messages.
          method: The `method` parameter in the `callback` function is an object that contains
        information about the received message. It includes properties such as the delivery tag, which is
        a unique identifier for the message, and other metadata like the exchange and routing key.
          properties: The "properties" parameter in the callback function represents the properties of
        the message being consumed. These properties can include metadata about the message, such as its
        content type, correlation ID, reply-to address, and more. The properties object allows you to
        access and manipulate these properties as needed.
          body: The `body` parameter in the `callback` function represents the message body that is
        received from the message queue. In this case, it is expected to be a string containing
        comma-separated float values.
        """
        # print(f"Process {self._rank}: Received: {body}")
        ch.basic_ack(delivery_tag=method.delivery_tag) 

        # Convert the received message to a NumPy array and store it
        try:
            received_data = np.fromstring(body, dtype=float, sep=',')
            self.messages_received.append(received_data)
        except ValueError as e:
            print(f"Failed to parse the received message: {e}")

        # Check if we have received the desired number of messages
        if len(self.messages_received) >=  self.length:
            ch.stop_consuming()

    def start_consuming(self, rank, itteration):
        """
        The function starts consuming messages and waits for data, printing the process rank and
        iteration.
        
        Args:
          rank: The "rank" parameter is used to identify the process or consumer. It is typically used in
        a distributed system where multiple processes or consumers are running concurrently. Each process
        or consumer is assigned a unique rank to differentiate them from each other.
          itteration: The `iteration` parameter is used to indicate the current iteration or round of the
        process. It is used in the print statement to display the current iteration number.
        """
        self._rank = rank
        print(f"Process {self._rank}: Waiting for data... in itteration {itteration}")

        # Set up a signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self.cleanup_and_exit)

        # Start consuming messages
        self.channel.start_consuming()

    def cleanup_and_exit(self, signal, frame):
        """
        The function `cleanup_and_exit` gracefully closes a connection and exits the script when Ctrl+C is
        pressed.
        
        Args:
          signal: The signal parameter represents the signal that triggered the cleanup_and_exit function.
        In this case, it is used to handle the Ctrl+C signal, which is sent when the user presses Ctrl+C on
        the keyboard.
          frame: The `frame` parameter is a reference to the current stack frame at the time the signal was
        received. It contains information about the current execution context, such as the code being
        executed and the local variables. In this case, the `frame` parameter is not used in the
        `cleanup_and_exit`
        """
        if self.connection and self.connection.is_open:
            self.connection.close()
        sys.exit(0)
    
    def memoryDown(self):
        """
        The function `memoryDown` use for  clean the data in `messages_received` array.
        """
        self.messages_received = []

    def get_received_messages(self):
        """
        Return the received messages as a list of NumPy arrays.
        """
        return self.messages_received




