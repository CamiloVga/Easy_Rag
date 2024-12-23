"""
Document loader module for handling different file formats.
"""
import os
import logging
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredFileLoader
)

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Class to manage document loading for different file formats."""
    
    SUPPORTED_FORMATS = [".pdf", ".docx", ".doc", ".csv", ".txt"]
    
    @staticmethod
    def load_file(file_path: str) -> List:
        """
        Load a single file based on its extension.
        
        Args:
            file_path (str): Path to the file to load
            
        Returns:
            List: List of loaded documents
            
        Raises:
            ValueError: If file format is not supported
            Exception: If there's an error loading the file
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in DocumentLoader.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {ext}")
        
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif ext == '.csv':
                loader = CSVLoader(file_path)
            else:  # fallback for txt and other text files
                loader = UnstructuredFileLoader(file_path)
            
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'title': os.path.basename(file_path),
                    'type': 'document',
                    'format': ext[1:],
                    'language': 'auto'
                })
            
            logger.info(f"Successfully loaded {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise