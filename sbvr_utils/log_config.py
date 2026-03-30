import logging
import colorlog
import os

def get_logger(name: str) -> logging.Logger:
    log_format = "%(log_color)s%(filename)s:%(lineno)d%(reset)s  %(message)s"
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        log_format,
        log_colors={
            "DEBUG": "blue",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
    ))

    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if already configured
    if not logger.handlers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    return logger


class ExtLogger:
    def __init__(self, log_path: str):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        
        self.log_path = os.path.join(parent_dir, "logs", log_path)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                pass
            
        self.COLOR_CODES = {
            "black": "\033[30m",
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
            "reset": "\033[0m"
        }
            
    
    def write(self, message: str, color: str = None):
        if color in self.COLOR_CODES:
            # Wrap the message in the given color and reset afterward
            colored_message = self.COLOR_CODES[color] + message + self.COLOR_CODES["reset"]
        else:
            colored_message = message
        with open(self.log_path, "a") as f:
            f.write(colored_message + "\n")