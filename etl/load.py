import os
from openai import OpenAI
import psycopg2
from dotenv import load_dotenv
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

db_conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    database=os.getenv("PGDATABASE"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
    port=os.getenv("PGPORT")
)

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def create_table():
    with db_conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS yongui_lore (
                id SERIAL PRIMARY KEY,
                chunk TEXT,
                embedding VECTOR(1536)
            );
        """)
        db_conn.commit()

def insert_chunk(chunk):
    embedding = embed_text(chunk)
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO yongui_lore (chunk, embedding) VALUES (%s, %s)",
            (chunk, embedding)
        )
        db_conn.commit()

def create_chunks(text, max_chunk_size=1000, overlap_size=200):
    # Split by parts first
    parts = re.split(r'Part [IVX]+:', text)
    
    chunks = []
    current_chunk = ""
    overlap_buffer = ""
    
    for part in parts:
        if not part.strip():
            continue
            
        # Split part into paragraphs
        paragraphs = part.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would exceed max size
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                
                # Create overlap from the end of the current chunk
                words = current_chunk.split()
                overlap_words = words[-int(overlap_size/5):]  # Approximate words for overlap
                overlap_buffer = " ".join(overlap_words)
                
                # Start new chunk with overlap
                current_chunk = overlap_buffer + "\n\n" + paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += paragraph
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = create_chunks(text)
    print(f"Created {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nProcessing chunk {i}/{len(chunks)}:")
        print(f"Length: {len(chunk)} characters")
        print(f"Preview: {chunk[:100]}...")
        if i > 1:
            print(f"Overlap with previous: {chunk[:50]}...")
        insert_chunk(chunk)

create_table()
process_file("etl/history.txt")
