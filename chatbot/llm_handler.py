"""
LLM handler for integrating with OpenAI and implementing RAG (Retrieval-Augmented Generation)
This module manages the conversation flow and response generation
"""

import logging
from typing import Dict, List, Optional, Any
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
import openai
import streamlit as st

logger = logging.getLogger(__name__)

class LLMHandler:
    """Handles LLM integration and RAG implementation"""
    
    def __init__(self, settings):
        """
        Initialize the LLM handler
        
        Args:
            settings: Configuration settings object
        """
        self.settings = settings
        self.llm = None
        self.memory = None
        self.qa_chain = None
        self.conversational_chain = None
        
        self._initialize_llm()
        self._initialize_memory()
        self._setup_prompts()
    
    def _initialize_llm(self):
        """Initialize the OpenAI language model"""
        try:
            # Initialize ChatOpenAI for better conversation handling
            self.llm = ChatOpenAI(
                model_name=self.settings.OPENAI_MODEL,
                temperature=self.settings.TEMPERATURE,
                max_tokens=self.settings.MAX_TOKENS,
                openai_api_key=self.settings.OPENAI_API_KEY,
                request_timeout=self.settings.REQUEST_TIMEOUT
            )
            
            logger.info(f"LLM initialized with model: {self.settings.OPENAI_MODEL}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise Exception(f"LLM initialization failed: {str(e)}")
    
    def _initialize_memory(self):
        """Initialize conversation memory"""
        try:
            self.memory = ConversationBufferWindowMemory(
                k=self.settings.MEMORY_WINDOW,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            logger.info("Conversation memory initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory: {str(e)}")
            raise Exception(f"Memory initialization failed: {str(e)}")
    
    def _setup_prompts(self):
        """Setup custom prompts for the chatbot"""
        # System prompt for FAQ assistant
        self.system_prompt = """You are a helpful and knowledgeable FAQ assistant for PT. Nawa Darsana Teknologi (Nawatech). 
        
        Your role is to:
        1. Answer questions about Nawatech's services, products, and company information
        2. Provide accurate and helpful information based on the provided context
        3. Be professional, friendly, and concise in your responses
        4. If you don't know something, admit it and suggest contacting Nawatech directly
        5. Always prioritize information from the provided FAQ context
        
        Company Information:
        - Company: PT. Nawa Darsana Teknologi (Nawatech)
        - Address: Gedung Office 8, Lantai 18 Unit A, SCBD Lot. 28, Jl Jend Sudirman Kav. 52-53, Senayan - Kebayoran Baru, Jakarta Selatan 12190
        - Phone: 021-29552754
        
        Guidelines:
        - Use the provided context to answer questions accurately
        - If the context doesn't contain relevant information, provide general guidance
        - Always maintain a professional and helpful tone
        - Keep responses concise but informative
        - If asked about services not in the FAQ, provide general information about Nawatech's technology capabilities
        """
        
        # QA prompt template
        self.qa_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            {system_prompt}
            
            Context from FAQ:
            {context}
            
            Question: {question}
            
            Answer: Provide a helpful and accurate response based on the context above. If the context doesn't contain relevant information, provide a general response about Nawatech and suggest contacting them directly for specific details.
            """.replace("{system_prompt}", self.system_prompt)
        )
        
        # Conversational prompt template
        self.conversational_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """
            Context from FAQ:
            {context}
            
            Chat History:
            {chat_history}
            
            Current Question: {question}
            
            Please provide a helpful response based on the context and conversation history.
            """)
        ])
        
        logger.info("Custom prompts setup completed")
    
    def generate_response(self, query: str, vector_store, use_conversation_history: bool = True) -> Dict[str, Any]:
        """
        Generate response using RAG approach
        
        Args:
            query: User query
            vector_store: FAISS vector store for document retrieval
            use_conversation_history: Whether to use conversation history
            
        Returns:
            Dictionary containing answer and metadata
        """
        try:
            # Retrieve relevant documents
            retrieved_docs = vector_store.similarity_search(
                query=query,
                k=self.settings.RETRIEVAL_K
            )
            
            # Prepare context from retrieved documents
            context = self._prepare_context(retrieved_docs)
            
            if use_conversation_history and self.memory:
                # Use conversational chain with memory
                response = self._generate_conversational_response(query, context)
            else:
                # Use simple QA without memory
                response = self._generate_qa_response(query, context)
            
            # Prepare response data
            response_data = {
                "answer": response,
                "source_documents": retrieved_docs,
                "context": context,
                "query": query
            }
            
            logger.info(f"Generated response for query: {query[:50]}...")
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "I apologize, but I'm experiencing technical difficulties. Please try again later or contact Nawatech directly at 021-29552754.",
                "source_documents": [],
                "context": "",
                "query": query,
                "error": str(e)
            }
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """
        Prepare context from retrieved documents
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No specific FAQ information found for this query."
        
        context_parts = []
        for i, doc in enumerate(documents):
            metadata = doc.metadata
            content = doc.page_content
            
            # Format document with metadata
            context_part = f"FAQ Item {i+1}:\n"
            if 'question' in metadata:
                context_part += f"Question: {metadata['question']}\n"
            if 'answer' in metadata:
                context_part += f"Answer: {metadata['answer']}\n"
            if 'category' in metadata:
                context_part += f"Category: {metadata['category']}\n"
            
            context_part += f"Content: {content}\n"
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _generate_qa_response(self, query: str, context: str) -> str:
        """
        Generate response using simple QA approach
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated response
        """
        try:
            # Format prompt
            formatted_prompt = self.qa_prompt_template.format(
                context=context,
                question=query
            )
            
            # Generate response
            response = self.llm.predict(formatted_prompt)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in QA response generation: {str(e)}")
            return "I apologize, but I couldn't process your question at the moment. Please try again."
    
    def _generate_conversational_response(self, query: str, context: str) -> str:
        """
        Generate response using conversational approach with memory
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated response
        """
        try:
            # Get chat history
            chat_history = self.memory.chat_memory.messages if self.memory else []
            
            # Format chat history for prompt
            formatted_history = ""
            for msg in chat_history[-6:]:  # Last 3 exchanges
                role = "Human" if msg.type == "human" else "Assistant"
                formatted_history += f"{role}: {msg.content}\n"
            
            # Create conversational chain if not exists
            if not self.conversational_chain:
                self.conversational_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=None,  # We handle retrieval manually
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=True
                )
            
            # Generate response using the conversational prompt
            messages = self.conversational_prompt.format_messages(
                context=context,
                chat_history=formatted_history,
                question=query
            )
            
            response = self.llm(messages)
            
            # Add to memory
            if self.memory:
                self.memory.chat_memory.add_user_message(query)
                self.memory.chat_memory.add_ai_message(response.content)
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error in conversational response generation: {str(e)}")
            return self._generate_qa_response(query, context)
    
    def clear_memory(self):
        """Clear conversation memory"""
        try:
            if self.memory:
                self.memory.clear()
                logger.info("Conversation memory cleared")
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
    
    def get_conversation_history(self) -> List[Dict]:
        """
        Get conversation history
        
        Returns:
            List of conversation messages
        """
        try:
            if not self.memory:
                return []
            
            messages = []
            for msg in self.memory.chat_memory.messages:
                messages.append({
                    "role": "human" if msg.type == "human" else "assistant",
                    "content": msg.content
                })
            
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []
    
    def validate_response_quality(self, query: str, response: str, context: str) -> Dict[str, float]:
        """
        Validate response quality using various metrics
        
        Args:
            query: Original query
            response: Generated response
            context: Retrieved context
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            metrics = {}
            
            # Response length check
            response_length = len(response.split())
            metrics['length_score'] = min(response_length / 50, 1.0)  # Optimal around 50 words
            
            # Context relevance (simple keyword matching)
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            context_words = set(context.lower().split())
            
            # Calculate overlap scores
            query_response_overlap = len(query_words & response_words) / max(len(query_words), 1)
            context_response_overlap = len(context_words & response_words) / max(len(context_words), 1)
            
            metrics['query_relevance'] = query_response_overlap
            metrics['context_usage'] = min(context_response_overlap * 2, 1.0)
            
            # Response completeness (checks for common incomplete patterns)
            incomplete_patterns = ['...', 'please contact', 'not sure', 'don\'t know']
            completeness_penalty = sum(1 for pattern in incomplete_patterns if pattern in response.lower())
            metrics['completeness'] = max(1.0 - completeness_penalty * 0.2, 0.0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error validating response quality: {str(e)}")
            return {
                'length_score': 0.5,
                'query_relevance': 0.5,
                'context_usage': 0.5,
                'completeness': 0.5
            }