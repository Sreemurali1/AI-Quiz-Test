import io
import logging
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load Environment Variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to extract text from PDF
def get_pdf_text(pdf: bytes) -> str:
    text = ""
    with io.BytesIO(pdf) as pdf_buffer:
        pdf_reader = PdfReader(pdf_buffer)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=350)
    chunks = text_splitter.split_text(text)
    return chunks

