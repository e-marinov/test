from confidence import CalibratedModel, load_pipeline
import pandas as pd
import pika
import json
import argparse
import logging
import time


class QueueWorker:
    def __init__(self, args):
        logging.info('Initializing worker.')
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=args.host[0],
                port=args.port[0],
                virtual_host=args.vhost[0],
            )
        )
        self.channel = self.connection.channel()
        self.channel.basic_qos(prefetch_count=1)
        self.model = load_pipeline(args.model[0], args.receipts[0])

    @property
    def queue_name(self):
        return 'data_engine.service'

    def start_loop(self):
        logging.info('Starting message processing loop.')
        parameters = {
            'x-dead-letter-exchange': '',
            'x-dead-letter-routing-key': 'data_engine.responder',
            'x-message-ttl': 10000,
        }
        self.channel.queue_declare(queue=self.queue_name, durable=True, arguments=parameters)
        self.channel.basic_consume(self.process_message, queue=self.queue_name)
        self.channel.start_consuming()

    def process_message(self, channel, method, properties, body):
        try:
            logging.debug('Received request: {}'.format(body))
            request = json.loads(body)
            method_name = '{}_callback'.format(request['action'])

            response = getattr(self, method_name)(request)
            logging.debug('Sending response: {}'.format(json.dumps(response)))
            channel.basic_publish(exchange='', body=json.dumps(response),
                                  routing_key=properties.reply_to)
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except StandardError as error:
            logging.error('Processing failed: {}'.format(error.message))
            channel.basic_publish(exchange='', body=body,
                                  routing_key=properties.reply_to)
            channel.basic_ack(delivery_tag=method.delivery_tag)

    def review_callback(self, request):
        response = request
        receipt = pd.DataFrame.from_records([request['receipt']], index='id')
        result = self.model.predict_proba(receipt)
        response['score'] = result[0]

        return response


def parse_arguments():
    parser = argparse.ArgumentParser(description='Data validation service.')

    parser.add_argument('--model', metavar='<file>', type=str,
                        nargs=1, default=['confidence_model.p'],
                        help='set the name of the model file to load; default: confidence_model.p')

    parser.add_argument('--receipts', metavar='<file>', type=str,
                        nargs=1, default=['receipts.csv'],
                        help='set the name of the reference receipts; default: receipts.csv')

    parser.add_argument('--host', metavar='HOST', type=str,
                        nargs=1, default=['localhost'],
                        help='address of the MQ server; default: localhost')

    parser.add_argument('--port', metavar='PORT', type=int,
                        nargs=1, default=[5672],
                        help='port of the MQ server; default: 5672')

    parser.add_argument('--vhost', metavar='NAME', type=str,
                        nargs=1, default=['/'],
                        help='virtual host to use on the MQ server; default: /')

    parser.add_argument('--log-level', metavar='LEVEL', type=str,
                        nargs=1, default=['INFO'],
                        help='set loglevel: DEBUG, INFO, WARNING, CRITICAL; default: INFO')

    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.log_level[0].upper()))

    while True:
        try:
            worker = QueueWorker(args)
            worker.start_loop()
        except Exception as error:
            logging.error('Worker failed: {}'.format(error.message))
            logging.info('Retrying in 60 seconds.')
            time.sleep(60)


if __name__ == "__main__":
    main()
