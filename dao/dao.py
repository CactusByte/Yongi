import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class YonguiDAO:
    _connection_pool = None

    @classmethod
    def initialize_pool(cls):
        if cls._connection_pool is None:
            try:
                cls._connection_pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    host=os.getenv("PGHOST"),
                    database=os.getenv("PGDATABASE"),
                    user=os.getenv("PGUSER"),
                    password=os.getenv("PGPASSWORD"),
                    port=os.getenv("PGPORT")
                )
                logger.info("Database connection pool initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing database connection pool: {str(e)}")
                raise

    def __init__(self):
        if YonguiDAO._connection_pool is None:
            YonguiDAO.initialize_pool()
        self.db_conn = YonguiDAO._connection_pool.getconn()

    def __del__(self):
        if hasattr(self, 'db_conn'):
            YonguiDAO._connection_pool.putconn(self.db_conn)

    def get_similar_chunks(self, vec_str: str, top_k: int):
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("BEGIN")
                try:
                    cur.execute(
                        """
                        SELECT chunk, 1 - (embedding <=> %s) as similarity
                        FROM yongui_lore
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (vec_str, vec_str, top_k)
                    )
                    results = cur.fetchall()
                    self.db_conn.commit()
                    return results
                except Exception as e:
                    self.db_conn.rollback()
                    logger.error(f"Database error in get_similar_chunks: {str(e)}")
                    raise
        except Exception as e:
            logger.error(f"Error in get_similar_chunks: {str(e)}")
            raise
