from src.data_loader import AnimeDataLoader
from src.vector_store import VectorStoreBuilder
from dotenv import load_dotenv

from config.paths_config import *
from utils.logger import get_logger
from utils.custom_exception import CustomException

load_dotenv()
logger = get_logger(__name__)


def main():
    try:

        logger.info("Starting to build pipeline.")

        loader = AnimeDataLoader(ORIGINAL_DATA_PATH, PROCSEED_DATA_PATH)
        processed_csv = loader.load_and_process()

        logger.info("Data loaded and processed successfully.")

        vector_store_builder = VectorStoreBuilder(processed_csv, PERSIST_DIR)
        vector_store_builder.build_and_save_vector_store()

        logger.info("Vector store built successfully.")

        logger.info("Pipeline built successfully.")
    

    except Exception as e:
            logger.info(f"Failed to built pipeline :  {str(e)}")
            raise CustomException("Error during building pipiline.", e)
    
    
    
if __name__ == "__main__":
     main()