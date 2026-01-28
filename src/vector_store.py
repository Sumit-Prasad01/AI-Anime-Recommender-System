from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings

from utils.custom_exception import CustomException
from utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)


class VectorStoreBuilder:
    def __init__(self, csv_path: str, persist_dir: str):

        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.embedding = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-V2"
        )

    def build_and_save_vector_store(self):
        try:
            loader = CSVLoader(
                file_path=self.csv_path,
                encoding="utf-8",
                metadata_columns=[]
            )

            documents = loader.load()

            splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=0
            )
            texts = splitter.split_documents(documents)

            db = Chroma.from_documents(
                documents=texts,
                embedding=self.embedding,
                persist_directory=self.persist_dir
            )

            logger.info("Vector store built and stored successfully.")

        except Exception as e:
            logger.error(f"Failed to build and store vector store - {e}")
            raise CustomException(
                "Error while building and storing vector store.",
                e
            )

    def load_vector_store(self):
        try:
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding
            )

        except Exception as e:
            logger.error(f"Failed to load vector store - {e}")
            raise CustomException(
                "Error while loading vector store.",
                e
            )
