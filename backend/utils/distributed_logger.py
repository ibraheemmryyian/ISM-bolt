import logging
import socket
import os
from logging.handlers import RotatingFileHandler, SocketHandler
from prometheus_client import Counter

class DistributedLogger:
    """Distributed, centralized logger for ML workflows (multi-process, multi-node)"""
    def __init__(self, name='DistributedLogger', log_file='logs/distributed.log', log_level=logging.INFO, elk_host=None, elk_port=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # File handler
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(process)d %(message)s'))
        self.logger.addHandler(file_handler)
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(console_handler)
        # ELK handler (optional)
        if elk_host and elk_port:
            try:
                elk_handler = SocketHandler(elk_host, elk_port)
                self.logger.addHandler(elk_handler)
            except Exception as e:
                self.logger.warning(f"Failed to connect to ELK: {e}")
        # Prometheus metrics
        self.log_counter = Counter('distributed_log_messages_total', 'Total log messages', ['level'])
    def info(self, msg):
        self.logger.info(msg)
        self.log_counter.labels(level='info').inc()
    def warning(self, msg):
        self.logger.warning(msg)
        self.log_counter.labels(level='warning').inc()
    def error(self, msg):
        self.logger.error(msg)
        self.log_counter.labels(level='error').inc()
    def debug(self, msg):
        self.logger.debug(msg)
        self.log_counter.labels(level='debug').inc()
    def critical(self, msg):
        self.logger.critical(msg)
        self.log_counter.labels(level='critical').inc()
    def get_logger(self):
        return self.logger 