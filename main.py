from utils import init_logger

if __name__ == "__main__":
    logger = init_logger("experiments/test/test.log")
    logger.info("test message")