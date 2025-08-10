"""
Data processor for handling FAQ data, preprocessing, and creating embeddings
This module handles Excel file processing, text chunking, and vector store creation
"""

import pandas as pd
import logging
import re
import os
from typing import List, Dict, Optional
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data processing, preprocessing, and vector store creation"""
    
    def __init__(self, settings):
        """
        Initialize the data processor
        
        Args:
            settings: Configuration settings object
        """
        self.settings = settings
        self.embeddings = None
        self.vector_store = None
        self.documents = []
        self.faq_data = None
        
        self._initialize_embeddings()
        self._load_and_process_data()
    
    def _initialize_embeddings(self):
        """Initialize OpenAI embeddings model"""
        try:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=self.settings.OPENAI_API_KEY,
                model="text-embedding-ada-002"  # Latest embedding model
            )
            logger.info("OpenAI embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise Exception(f"Embeddings initialization failed: {str(e)}")
    
    def _load_and_process_data(self):
        """Load FAQ data from Excel file and process it"""
        try:
            # Load FAQ data from Excel file
            faq_file_path = os.path.join("data", "FAQ_Nawa.xlsx")
            
            if not os.path.exists(faq_file_path):
                # Create sample data if file doesn't exist
                logger.warning(f"FAQ file not found at {faq_file_path}. Creating sample data.")
                self._create_sample_faq_data()
            else:
                self.faq_data = pd.read_excel(faq_file_path)
                logger.info(f"Loaded FAQ data with {len(self.faq_data)} entries")
            
            # Process the data
            self._preprocess_data()
            self._create_documents()
            self._create_vector_store()
            
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise Exception(f"Data processing failed: {str(e)}")
    
    def _create_sample_faq_data(self):
        """Create sample FAQ data if the original file is not available"""
        sample_data = {
            'Question': [
                'What is Nawatech?',
                'What services does Nawatech provide?',
                'How can I contact Nawatech?',
                'Where is Nawatech located?',
                'What are Nawatech business hours?',
                'Does Nawatech provide cloud services?',
                'What is Nawatech expertise in machine learning?',
                'How does Nawatech ensure data security?',
                'What industries does Nawatech serve?',
                'How can I get a quote for services?'
            ],
            'Answer': [
                'PT. Nawa Darsana Teknologi (Nawatech) is a technology company that provides innovative solutions in software development, cloud services, and digital transformation.',
                'Nawatech provides software development, cloud solutions, machine learning services, data analytics, and digital transformation consulting.',
                'You can contact Nawatech by phone at 021-29552754 or visit our office at Gedung Office 8, Lantai 18 Unit A, SCBD Lot. 28, Jakarta Selatan.',
                'Nawatech is located at Gedung Office 8, Lantai 18 Unit A, SCBD Lot. 28, Jl Jend Sudirman Kav. 52-53, Senayan - Kebayoran Baru, Jakarta Selatan 12190.',
                'Our business hours are Monday to Friday, 9:00 AM to 6:00 PM Western Indonesia Time (WIB).',
                'Yes, Nawatech provides comprehensive cloud services including cloud migration, infrastructure management, and cloud-native application development.',
                'Nawatech has extensive expertise in machine learning, including natural language processing, computer vision, predictive analytics, and AI model deployment.',
                'Nawatech implements industry-standard security practices including encryption, access controls, regular security audits, and compliance with data protection regulations.',
                'Nawatech serves various industries including finance, healthcare, retail, manufacturing, and government sectors.',
                'To get a quote, please contact us at 021-29552754 or send an email with your requirements. Our team will provide a detailed proposal within 2-3 business days.'
            ],
            'Category': [
                'Company Info',
                'Services',
                'Contact',
                'Location',
                'General',
                'Services',
                'Technology',
                'Security',
                'Business',
                'Sales'
            ]
        }
        
        self.faq_data = pd.DataFrame(sample_data)
        logger.info("Created sample FAQ data with 10 entries")
    
    def _preprocess_data(self):
        """Preprocess the FAQ data for better search and retrieval"""
        if self.faq_data is None:
            raise Exception("No FAQ data available for preprocessing")
        
        # Clean and normalize text
        self.faq_data['Question_Clean'] = self.faq_data['Question'].apply(self._clean_text)
        self.faq_data['Answer_Clean'] = self.faq_data['Answer'].apply(self._clean_text)
        
        # Create combined text for better context
        self.faq_data['Combined_Text'] = (
            "Question: " + self.faq_data['Question_Clean'] + 
            "\nAnswer: " + self.faq_data['Answer_Clean']
        )
        
        # Add metadata if Category column exists
        if 'Category' in self.faq_data.columns:
            self.faq_data['Category_Clean'] = self.faq_data['Category'].apply(self._clean_text)
        else:
            self.faq_data['Category_Clean'] = 'General'
        
        logger.info("Data preprocessing completed successfully")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text data
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove extra whitespaces and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text
    
    def _create_documents(self):
        """Create LangChain documents from processed FAQ data"""
        self.documents = []
        
        # Text splitter for handling long documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        for idx, row in self.faq_data.iterrows():
            # Create document with question and answer
            content = row['Combined_Text']
            
            # Split content if it's too long
            chunks = text_splitter.split_text(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                metadata = {
                    'source': f'FAQ_{idx}_{chunk_idx}',
                    'question': row['Question_Clean'],
                    'answer': row['Answer_Clean'],
                    'category': row['Category_Clean'],
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks)
                }
                
                document = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                
                self.documents.append(document)
        
        logger.info(f"Created {len(self.documents)} document chunks from FAQ data")
    
    def _create_vector_store(self):
        """Create FAISS vector store from documents"""
        try:
            if not self.documents:
                raise Exception("No documents available for vector store creation")
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                documents=self.documents,
                embedding=self.embeddings
            )
            
            logger.info(f"FAISS vector store created with {len(self.documents)} documents")
            
            # Save vector store for future use (optional)
            self._save_vector_store()
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise Exception(f"Vector store creation failed: {str(e)}")
    
    def _save_vector_store(self):
        """Save vector store to disk for persistence (optional)"""
        try:
            vector_store_path = os.path.join("data", "vector_store")
            os.makedirs(vector_store_path, exist_ok=True)
            
            self.vector_store.save_local(vector_store_path)
            logger.info(f"Vector store saved to {vector_store_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save vector store: {str(e)}")
    
    def get_similar_documents(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve similar documents for a given query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of similar documents
        """
        try:
            if not self.vector_store:
                raise Exception("Vector store not initialized")
            
            # Perform similarity search
            similar_docs = self.vector_store.similarity_search(
                query=query,
                k=k
            )
            
            logger.info(f"Retrieved {len(similar_docs)} similar documents for query: {query[:50]}...")
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error retrieving similar documents: {str(e)}")
            return []
    
    def get_similar_documents_with_scores(self, query: str, k: int = 4) -> List[tuple]:
        """
        Retrieve similar documents with similarity scores
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of tuples (document, score)
        """
        try:
            if not self.vector_store:
                raise Exception("Vector store not initialized")
            
            # Perform similarity search with scores
            similar_docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            logger.info(f"Retrieved {len(similar_docs_with_scores)} documents with scores")
            
            return similar_docs_with_scores
            
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {str(e)}")
            return []
    
    def get_faq_statistics(self) -> Dict:
        """
        Get statistics about the FAQ data
        
        Returns:
            Dictionary with FAQ statistics
        """
        if self.faq_data is None:
            return {}
        
        stats = {
            'total_faqs': len(self.faq_data),
            'total_documents': len(self.documents),
            'categories': self.faq_data['Category_Clean'].unique().tolist() if 'Category_Clean' in self.faq_data.columns else [],
            'avg_question_length': self.faq_data['Question_Clean'].str.len().mean() if 'Question_Clean' in self.faq_data.columns else 0,
            'avg_answer_length': self.faq_data['Answer_Clean'].str.len().mean() if 'Answer_Clean' in self.faq_data.columns else 0
        }
        
        return stats