"""Data Exporters for various file formats."""

from abc import ABC, abstractmethod
from typing import Iterator, Optional

import pandas as pd


class BaseExporter(ABC):
    """Abstract base class for data exporters."""

    @abstractmethod
    def export(self, data: pd.DataFrame, path: str, **kwargs) -> None:
        """Export data to file.

        Args:
            data: DataFrame to export.
            path: Path to the output file.
            **kwargs: Additional arguments passed to the underlying writer.
        """
        raise NotImplementedError

    def export_stream(self, data_iter: Iterator[pd.DataFrame], path: str, **kwargs) -> None:
        """Export data in chunks from an iterator.

        Args:
            data_iter: Iterator of DataFrames to export.
            path: Path to the output file.
            **kwargs: Additional arguments passed to the underlying writer.
        """
        raise NotImplementedError("Streaming not supported for this exporter")


class CSVExporter(BaseExporter):
    """Exporter for CSV files."""

    def export(self, data: pd.DataFrame, path: str, **kwargs) -> None:
        """Export data to CSV file.

        Args:
            data: DataFrame to export.
            path: Path to the output CSV file.
            **kwargs: Additional arguments passed to DataFrame.to_csv.
        """
        data.to_csv(path, index=False, **kwargs)

    def export_stream(self, data_iter: Iterator[pd.DataFrame], path: str, **kwargs) -> None:
        """Export data to CSV file in chunks.

        Args:
            data_iter: Iterator of DataFrames to export.
            path: Path to the output CSV file.
            **kwargs: Additional arguments passed to DataFrame.to_csv.
        """
        index = kwargs.pop('index', False)
        write_header = True
        mode = 'w'

        for i, df in enumerate(data_iter):
            if i == 0:
                df.to_csv(path, index=index, mode=mode, header=write_header, **kwargs)
            else:
                df.to_csv(path, index=index, mode='a', header=False, **kwargs)


class JSONExporter(BaseExporter):
    """Exporter for JSON files."""

    def export(self, data: pd.DataFrame, path: str, **kwargs) -> None:
        """Export data to JSON file.

        Args:
            data: DataFrame to export.
            path: Path to the output JSON file.
            **kwargs: Additional arguments passed to DataFrame.to_json.
        """
        data.to_json(path, **kwargs)

    def export_stream(self, data_iter: Iterator[pd.DataFrame], path: str, **kwargs) -> None:
        """Export data to JSON file in chunks (JSON Lines format).

        Args:
            data_iter: Iterator of DataFrames to export.
            path: Path to the output JSON file.
            **kwargs: Additional arguments passed to DataFrame.to_json.
        """
        orient = kwargs.pop('orient', 'records')

        with open(path, 'w') as f:
            for df in data_iter:
                for record in df.to_dict(orient=orient):
                    import json
                    f.write(json.dumps(record) + '\n')
