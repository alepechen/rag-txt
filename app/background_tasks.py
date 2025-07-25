from sqlalchemy.orm import Session
from db import FileChunk
import nltk
from nltk.tokenize import sent_tokenize
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
nltk.download('punkt')

class TextProcesser:
    def __init__(self, db: Session, file_id: int, chunk_size: int = 2):
        self.db = db
        self.file_id = file_id
        self.chunk_size = chunk_size
    
    def chunk_and_embed(self, text: str):
        sentences = sent_tokenize(text)
        chunks = [' '.join(sentences[i:i + self.chunk_size])
                for i in range(0, len(sentences), self.chunk_size)]
        for chunk in chunks:
            response = client.embeddings.create(
                model="text-embedding-ada-002", 
                input=chunk
            )
            embedding_vector = response.data[0].embedding
            file_chunk = FileChunk(file_id=self.file_id,
                                chunk_text=chunk,
                                embedding_vector=embeddings)
            self.db.add(file_chunk)
        self.db.commit()