import logging

# Console logger configuration
console_logger = logging.getLogger('console')
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
console_logger.addHandler(console_handler)
console_logger.setLevel(logging.INFO)


def create_file_loger(file_name: str):
    """Creates and configures a file logger with the specified name.

        Args:
            file_name (str): Name of the log file to be created.

        Returns:
            logging.Logger: Configured file logger instance that writes to /app/files/logs/{file_name}
                with INFO level and timestamp formatting.

        Example:
            logger = create_file_loger("process.log")
            logger.info("Processing started")
        """
    file_logger = logging.getLogger(file_name)
    file_handler = logging.FileHandler(f'/app/files/logs/{file_name}')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_logger.addHandler(file_handler)
    file_logger.setLevel(logging.INFO)

    return file_logger
