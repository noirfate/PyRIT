import logging
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import MutableSequence, Optional, Sequence, TypeVar, Union, Literal

from sqlalchemy import MetaData, create_engine, text
from sqlalchemy.engine.base import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload, sessionmaker
from sqlalchemy.orm.session import Session

from pyrit.common import default_values
from pyrit.common.singleton import Singleton
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.memory.memory_models import Base, EmbeddingDataEntry, PromptMemoryEntry
from pyrit.models import PromptRequestPiece

logger = logging.getLogger(__name__)

Model = TypeVar("Model")

# 定义支持的数据库类型
DbType = Literal["postgresql", "mysql", "sqlserver"]

class RdsSQLMemory(MemoryInterface, metaclass=Singleton):
    """
    A class to manage conversation memory using relational database as the backend.
    It supports PostgreSQL, MySQL and SQL Server.
    
    It leverages SQLAlchemy Base models for creating tables and provides CRUD 
    operations to interact with the tables.

    This class encapsulates the setup of the database connection, table creation 
    based on SQLAlchemy models, and session management to perform database operations.
    """

    def __init__(
        self,
        *,
        connection_string: Optional[str] = None,
        db_type: DbType = "postgresql",
        verbose: bool = False,
    ):
        """
        Initialize the RdsSQLMemory instance.
        
        Args:
            connection_string: Database connection string
            db_type: Database type, supporting "postgresql", "mysql", or "sqlserver"
            verbose: Whether to enable verbose logging
        """
        self._db_type = db_type
        
        # 所有数据库类型统一使用RDS_CONNECTION_STRING环境变量
        # 连接字符串格式示例:
        # PostgreSQL: postgresql://username:password@hostname:port/database_name
        # MySQL: mysql://username:password@hostname:port/database_name
        # SQL Server: mssql+pyodbc://username:password@server/database?driver=ODBC+Driver+17+for+SQL+Server
        
        self._connection_string = default_values.get_required_value(
            env_var_name="RDS_CONNECTION_STRING", passed_value=connection_string
        )

        self.engine = self._create_engine(has_echo=verbose)
        self.SessionFactory = sessionmaker(bind=self.engine)
        self._create_tables_if_not_exist()

        super().__init__()
        
    def _init_storage_io(self):
        """
        Initialize storage IO.
        This is a required method from MemoryInterface but not needed for RDS implementation.
        """
        # RDS implementation doesn't require external storage for results
        pass

    def _create_engine(self, *, has_echo: bool) -> Engine:
        """Creates the SQLAlchemy engine for the specified database type.

        Creates an engine bound to the specified server and database. The `has_echo` parameter
        controls the verbosity of SQL execution logging.

        Args:
            has_echo (bool): Flag to enable detailed SQL execution logging.
        """

        try:
            # Create the SQLAlchemy engine.
            # Use pool_pre_ping (health check) to gracefully handle server-closed connections
            # by testing and replacing stale connections.
            # Set pool_recycle to 1800 seconds to prevent connections from being closed due to server timeout.
            engine = create_engine(
                self._connection_string,
                pool_recycle=1800,
                pool_pre_ping=True,
                echo=has_echo,
                pool_size=5,
                max_overflow=10
            )
            logger.info(f"Engine created successfully for database: {engine.name} (type: {self._db_type})")
            return engine
        except SQLAlchemyError as e:
            logger.exception(f"Error creating the engine for the database: {e}")
            raise

    def _create_tables_if_not_exist(self):
        """
        Creates all tables defined in the Base metadata, if they don't already exist in the database.

        Raises:
            Exception: If there's an issue creating the tables in the database.
        """
        try:
            # Using the 'checkfirst=True' parameter to avoid attempting to recreate existing tables
            Base.metadata.create_all(self.engine, checkfirst=True)
        except Exception as e:
            logger.error(f"Error during table creation: {e}")

    def _add_embeddings_to_memory(self, *, embedding_data: Sequence[EmbeddingDataEntry]) -> None:
        """
        Inserts embedding data into memory storage
        """
        self._insert_entries(entries=embedding_data)

    def _get_prompt_pieces_memory_label_conditions(self, *, memory_labels: dict[str, str]):
        """
        Generate SQL conditions for querying based on memory labels.
        Different database types have different JSON query syntax.
        """
        if self._db_type == "postgresql":
            # PostgreSQL JSON syntax
            json_validation = "labels::jsonb IS NOT NULL"
            json_conditions = " AND ".join([f"labels->>'{key}' = :{key}" for key in memory_labels])
        elif self._db_type == "mysql":
            # MySQL JSON syntax
            json_validation = "labels IS NOT NULL"
            json_conditions = " AND ".join([f"JSON_UNQUOTE(JSON_EXTRACT(labels, '$.{key}')) = :{key}" for key in memory_labels])
        else:  # sqlserver
            # SQL Server JSON syntax
            json_validation = "ISJSON(labels) = 1"
            json_conditions = " AND ".join([f"JSON_VALUE(labels, '$.{key}') = :{key}" for key in memory_labels])
            
        # Combine both conditions
        conditions = f"{json_validation} AND {json_conditions}"

        # Create SQL condition using SQLAlchemy's text() with bindparams
        # for safe parameter passing, preventing SQL injection
        return text(conditions).bindparams(**{key: str(value) for key, value in memory_labels.items()})

    def _get_prompt_pieces_orchestrator_conditions(self, *, orchestrator_id: str):
        """
        Generate SQL conditions for querying based on orchestrator ID.
        Different database types have different JSON query syntax.
        """
        if self._db_type == "postgresql":
            # PostgreSQL JSON syntax
            conditions = "orchestrator_identifier::jsonb IS NOT NULL AND orchestrator_identifier->>'id' = :json_id"
        elif self._db_type == "mysql":
            # MySQL JSON syntax
            conditions = "orchestrator_identifier IS NOT NULL AND JSON_UNQUOTE(JSON_EXTRACT(orchestrator_identifier, '$.id')) = :json_id"
        else:  # sqlserver
            # SQL Server JSON syntax
            conditions = "ISJSON(orchestrator_identifier) = 1 AND JSON_VALUE(orchestrator_identifier, '$.id') = :json_id"
            
        return text(conditions).bindparams(json_id=str(orchestrator_id))

    def _get_metadata_conditions(self, *, prompt_metadata: dict[str, Union[str, int]]):
        """
        Generate SQL conditions for querying based on prompt metadata.
        Different database types have different JSON query syntax.
        """
        if self._db_type == "postgresql":
            # PostgreSQL JSON syntax
            json_validation = "prompt_metadata::jsonb IS NOT NULL"
            json_conditions = " AND ".join([f"prompt_metadata->>'{key}' = :{key}" for key in prompt_metadata])
        elif self._db_type == "mysql":
            # MySQL JSON syntax
            json_validation = "prompt_metadata IS NOT NULL"
            json_conditions = " AND ".join([f"JSON_UNQUOTE(JSON_EXTRACT(prompt_metadata, '$.{key}')) = :{key}" for key in prompt_metadata])
        else:  # sqlserver
            # SQL Server JSON syntax
            json_validation = "ISJSON(prompt_metadata) = 1"
            json_conditions = " AND ".join([f"JSON_VALUE(prompt_metadata, '$.{key}') = :{key}" for key in prompt_metadata])
            
        # Combine both conditions
        conditions = f"{json_validation} AND {json_conditions}"

        # Create SQL condition using SQLAlchemy's text() with bindparams
        # for safe parameter passing, preventing SQL injection
        return text(conditions).bindparams(**{key: str(value) for key, value in prompt_metadata.items()})

    def _get_prompt_pieces_prompt_metadata_conditions(self, *, prompt_metadata: dict[str, Union[str, int]]):
        return self._get_metadata_conditions(prompt_metadata=prompt_metadata)

    def _get_seed_prompts_metadata_conditions(self, *, metadata: dict[str, Union[str, int]]):
        return self._get_metadata_conditions(prompt_metadata=metadata)

    def add_request_pieces_to_memory(self, *, request_pieces: Sequence[PromptRequestPiece]) -> None:
        """
        Inserts a list of prompt request pieces into the memory storage.
        """
        self._insert_entries(entries=[PromptMemoryEntry(entry=piece) for piece in request_pieces])

    def dispose_engine(self):
        """
        Dispose the engine and clean up resources.
        """
        if self.engine:
            self.engine.dispose()
            logger.info("Engine disposed successfully.")

    def get_all_embeddings(self) -> Sequence[EmbeddingDataEntry]:
        """
        Fetches all entries from the specified table and returns them as model instances.
        """
        result: Sequence[EmbeddingDataEntry] = self._query_entries(EmbeddingDataEntry)
        return result

    def _insert_entry(self, entry: Base) -> None:  # type: ignore
        """
        Inserts an entry into the Table.

        Args:
            entry: An instance of a SQLAlchemy model to be added to the Table.
        """
        with closing(self.get_session()) as session:
            try:
                session.add(entry)
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                logger.exception(f"Error inserting entry into the table: {e}")

    def _insert_entries(self, *, entries: Sequence[Base]) -> None:  # type: ignore
        """Inserts multiple entries into the database."""
        with closing(self.get_session()) as session:
            try:
                session.add_all(entries)
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                logger.exception(f"Error inserting multiple entries into the table: {e}")
                raise

    def get_session(self) -> Session:
        """
        Provides a session for database operations.
        """
        return self.SessionFactory()

    def _query_entries(
        self, Model, *, conditions: Optional = None, distinct: bool = False, join_scores: bool = False  # type: ignore
    ) -> MutableSequence[Model]:
        """
        Fetches data from the specified table model with optional conditions.

        Args:
            model: The SQLAlchemy model class corresponding to the table you want to query.
            conditions: SQLAlchemy filter conditions (Optional).
            distinct: Flag to return distinct rows (defaults to False).
            join_scores: Flag to join the scores table with entries (defaults to False).

        Returns:
            List of model instances representing the rows fetched from the table.
        """
        with closing(self.get_session()) as session:
            try:
                query = session.query(Model)
                if join_scores and Model == PromptMemoryEntry:
                    query = query.options(joinedload(PromptMemoryEntry.scores))
                if conditions is not None:
                    query = query.filter(conditions)
                if distinct:
                    return query.distinct().all()
                return query.all()
            except SQLAlchemyError as e:
                logger.exception(f"Error fetching data from table {Model.__tablename__}: {e}")
                return []

    def _update_entries(self, *, entries: MutableSequence[Base], update_fields: dict) -> bool:  # type: ignore
        """
        Updates the given entries with the specified field values.

        Args:
            entries (Sequence[Base]): A list of SQLAlchemy model instances to be updated.
            update_fields (dict): A dictionary of field names and their new values.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        if not update_fields:
            raise ValueError("update_fields must be provided to update prompt entries.")
        with closing(self.get_session()) as session:
            try:
                for entry in entries:
                    # Ensure the entry is attached to the session. If it's detached, merge it.
                    if not session.is_modified(entry):
                        entry_in_session = session.merge(entry)
                    else:
                        entry_in_session = entry
                    for field, value in update_fields.items():
                        if field in vars(entry_in_session):
                            setattr(entry_in_session, field, value)
                        else:
                            session.rollback()
                            raise ValueError(
                                f"Field '{field}' does not exist in the table \
                                            '{entry_in_session.__tablename__}'. Rolling back changes..."
                            )
                session.commit()
                return True
            except SQLAlchemyError as e:
                session.rollback()
                logger.exception(f"Error updating entries: {e}")
                return False

    def reset_database(self):
        """Drop and recreate existing tables"""
        # Drop all existing tables
        Base.metadata.drop_all(self.engine)
        # Recreate the tables
        Base.metadata.create_all(self.engine, checkfirst=True)

    def print_schema(self):
        """Prints the schema of all tables in the database."""
        metadata = MetaData()
        metadata.reflect(bind=self.engine)

        for table_name in metadata.tables:
            table = metadata.tables[table_name]
            print(f"Schema for {table_name}:")
            for column in table.columns:
                print(f"  Column {column.name} ({column.type})")
