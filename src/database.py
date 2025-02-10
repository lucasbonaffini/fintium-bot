from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import logging

class Database:
    def __init__(self, config):
        try:
            # Construir connection string
            connection_string = (
                f"postgresql://{config.user}:{config.password}@"
                f"{config.host}:{config.port}/{config.database}"
            )
            
            # Crear engine con parámetros optimizados
            self.engine = create_engine(
                connection_string,
                echo=False,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30
            )
            
            self.SessionLocal = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
            
            # Verificar conexión
            self.test_connection()
            logging.info("Database connection established successfully")
            
        except Exception as e:
            logging.error(f"Database connection error: {str(e)}")
            raise

    def test_connection(self):
        """Test database connection"""
        try:
            with self.get_session() as session:
                # Usar text() para la consulta SQL
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logging.error(f"Database connection test failed: {str(e)}")
            return False

    def get_session(self):
        """Get database session with proper error handling"""
        try:
            session = self.SessionLocal()
            return session
        except Exception as e:
            logging.error(f"Error creating database session: {str(e)}")
            raise

    def create_tables(self):
        """Create database tables"""
        try:
            from base import Base
            Base.metadata.create_all(self.engine)
            logging.info("Database tables created successfully")
        except Exception as e:
            logging.error(f"Error creating tables: {str(e)}")
            raise