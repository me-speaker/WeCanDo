"""Data Importers for various file formats."""

from abc import ABC, abstractmethod
from typing import Iterator, Optional

import pandas as pd


class BaseImporter(ABC):
    """Abstract base class for data importers."""

    @abstractmethod
    def import_data(self, path: str, **kwargs) -> pd.DataFrame:
        """Import data from file.

        Args:
            path: Path to the data file.
            **kwargs: Additional arguments passed to the underlying reader.

        Returns:
            DataFrame containing the imported data.
        """
        raise NotImplementedError

    def import_data_stream(self, path: str, chunk_size: int = 10000, **kwargs) -> Iterator[pd.DataFrame]:
        """Import data in chunks for large files.

        Args:
            path: Path to the data file.
            chunk_size: Number of rows per chunk.
            **kwargs: Additional arguments passed to the underlying reader.

        Yields:
            DataFrame chunks.
        """
        raise NotImplementedError("Streaming not supported for this importer")


class CSVImporter(BaseImporter):
    """Importer for CSV files."""

    def import_data(self, path: str, **kwargs) -> pd.DataFrame:
        """Import data from CSV file.

        Args:
            path: Path to the CSV file.
            **kwargs: Additional arguments passed to pd.read_csv.

        Returns:
            DataFrame containing the imported data.
        """
        return pd.read_csv(path, **kwargs)

    def import_data_stream(self, path: str, chunk_size: int = 10000, **kwargs) -> Iterator[pd.DataFrame]:
        """Import CSV data in chunks.

        Args:
            path: Path to the CSV file.
            chunk_size: Number of rows per chunk.
            **kwargs: Additional arguments passed to pd.read_csv.

        Yields:
            DataFrame chunks.
        """
        for chunk in pd.read_csv(path, chunksize=chunk_size, **kwargs):
            yield chunk


class JSONImporter(BaseImporter):
    """Importer for JSON files."""

    def import_data(self, path: str, **kwargs) -> pd.DataFrame:
        """Import data from JSON file.

        Args:
            path: Path to the JSON file.
            **kwargs: Additional arguments passed to pd.read_json.

        Returns:
            DataFrame containing the imported data.
        """
        return pd.read_json(path, **kwargs)

    def import_data_stream(self, path: str, chunk_size: int = 10000, **kwargs) -> Iterator[pd.DataFrame]:
        """Import JSON data in chunks (for JSON Lines format).

        Args:
            path: Path to the JSON file.
            chunk_size: Number of rows per chunk.
            **kwargs: Additional arguments passed to pd.read_json.

        Yields:
            DataFrame chunks.
        """
        chunks = []
        chunk_count = 0
        with open(path, 'r', **kwargs.get('open_kwargs', {})) as f:
            for line in f:
                import json
                chunks.append(json.loads(line))
                if len(chunks) >= chunk_size:
                    chunk_count += 1
                    yield pd.DataFrame(chunks)
                    chunks = []
        if chunks:
            yield pd.DataFrame(chunks)


class SQLiteImporter(BaseImporter):
    """Importer for SQLite databases."""

    def import_data(self, path: str, query: str, **kwargs) -> pd.DataFrame:
        """Import data from SQLite database.

        Args:
            path: Path to the SQLite database file.
            query: SQL query to execute.
            **kwargs: Additional arguments passed to pd.read_sql_query.

        Returns:
            DataFrame containing the query results.
        """
        import sqlite3
        conn = sqlite3.connect(path)
        try:
            return pd.read_sql_query(query, conn, **kwargs)
        finally:
            conn.close()

    def import_data_stream(self, path: str, query: str, chunk_size: int = 10000, **kwargs) -> Iterator[pd.DataFrame]:
        """Import SQLite data in chunks.

        Args:
            path: Path to the SQLite database file.
            query: SQL query to execute.
            chunk_size: Number of rows per chunk.
            **kwargs: Additional arguments passed to pd.read_sql_query.

        Yields:
            DataFrame chunks.
        """
        import sqlite3
        conn = sqlite3.connect(path)
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [description[0] for description in cursor.description]

            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                yield pd.DataFrame(rows, columns=columns)
        finally:
            conn.close()


class ExcelImporter(BaseImporter):
    """Importer for Excel files (.xlsx, .xls)."""

    def import_data(self, path: str, **kwargs) -> pd.DataFrame:
        """Import data from Excel file.

        Args:
            path: Path to the Excel file.
            **kwargs: Additional arguments passed to pd.read_excel.
                - sheet_name: Sheet name or index (default: 0)
                - header: Row number to use as header (default: 0)

        Returns:
            DataFrame containing the imported data.
        """
        return pd.read_excel(path, **kwargs)

    def import_data_stream(self, path: str, chunk_size: int = 10000, **kwargs) -> Iterator[pd.DataFrame]:
        """Import Excel data in chunks.

        Note: Excel files are not naturally chunked like CSV.
        This method reads all sheets and yields them if they have enough rows.

        Args:
            path: Path to the Excel file.
            chunk_size: Minimum rows per chunk (default: 10000).
            **kwargs: Additional arguments passed to pd.read_excel.

        Yields:
            DataFrame chunks (one sheet at a time for large sheets).
        """
        # For Excel, we read all sheets and yield them
        # If a single sheet has more than chunk_size rows, we could chunk it further
        # but pd.read_excel doesn't support streaming, so we yield whole sheets
        sheet_name = kwargs.pop('sheet_name', 0)
        df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)

        # If the dataframe is large enough, split it
        if len(df) > chunk_size:
            for i in range(0, len(df), chunk_size):
                yield df.iloc[i:i + chunk_size]
        else:
            yield df