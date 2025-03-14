import logging
import os
from datetime import datetime

# Ensure the logs directory exists
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Create a log file with a timestamp
log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_file_path = os.path.join(logs_path, log_file)

# Configure logging for the root logger
logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO  # Ensure it's set to INFO level to capture info logs
)

# Create a logger instance for your custom logging
logger = logging.getLogger(__name__)

# Set log level for other loggers (such as werkzeug) if needed
logging.getLogger('werkzeug').setLevel(logging.WARNING)  # To only log warnings or errors for werkzeug
