import os
import sys
import logging
from datetime import datetime

logging_str = "[%(asctime)s - %(levelname)s - %(module)s - %(message)s]"

LOG_FILE = f"{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}.log"
log_file_path = os.path.join(os.getcwd(),"LOGS",LOG_FILE)
os.makedirs(log_file_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_file_path, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
        
    ]
)

logger = logging.getLogger("cnnClassifierLogger")
