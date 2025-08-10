"""
Main Streamlit application for Nawatech FAQ Chatbot
This file handles the UI and orchestrates the chatbot functionality
"""

import streamlit as st
import logging
import time
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import custom modules
from chatbot.data_processor import DataProcessor
from chatbot.llm_handler import LLMHandler
from chatbot.security import SecurityManager
from chatbot.quality_scorer import QualityScorer
from config.settings import Settings

# Page configuration
st.set_page_config(
    page_title="Nawatech FAQ Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ChatbotApp:
    """Main chatbot application class"""
    
    def __init__(self):
        """Initialize the chatbot application"""
        self.settings = Settings()
        self.security_manager = SecurityManager()
        self.data_processor = None
        self.llm_handler = None
        self.quality_scorer = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all chatbot components"""
        try:
            # Initialize session state
            self._init_session_state()
            
            # Initialize components with caching
            if 'data_processor' not in st.session_state:
                with st.spinner("Loading FAQ data..."):
                    self.data_processor = DataProcessor(self.settings)
                    st.session_state.data_processor = self.data_processor
                    logger.info("Data processor initialized successfully")
            else:
                self.data_processor = st.session_state.data_processor
            
            if 'llm_handler' not in st.session_state:
                with st.spinner("Initializing AI model..."):
                    self.llm_handler = LLMHandler(self.settings)
                    st.session_state.llm_handler = self.llm_handler
                    logger.info("LLM handler initialized successfully")
            else:
                self.llm_handler = st.session_state.llm_handler
            
            if 'quality_scorer' not in st.session_state:
                self.quality_scorer = QualityScorer()
                st.session_state.quality_scorer = self.quality_scorer
                logger.info("Quality scorer initialized successfully")
            else:
                self.quality_scorer = st.session_state.quality_scorer
                
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            st.error(f"Failed to initialize chatbot: {str(e)}")
            st.stop()
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'user_id' not in st.session_state:
            st.session_state.user_id = f"user_{int(time.time())}"
        
        if 'conversation_started' not in st.session_state:
            st.session_state.conversation_started = False
        
        if 'total_queries' not in st.session_state:
            st.session_state.total_queries = 0
        
        if 'quality_scores' not in st.session_state:
            st.session_state.quality_scores = []
    
    def _render_sidebar(self):
        """Render the sidebar with chatbot information and controls"""
        with st.sidebar:
            st.image("https://via.placeholder.com/200x100/1f77b4/white?text=Nawatech", 
                    caption="Nawatech FAQ Assistant")
            
            st.markdown("### 🤖 Chatbot Information")
            st.info("""
            This AI assistant can help you with:
            - Company information
            - Services and products
            - Technical support
            - General inquiries about Nawatech
            """)
            
            # Statistics
            if st.session_state.total_queries > 0:
                st.markdown("### 📊 Session Statistics")
                st.metric("Queries Asked", st.session_state.total_queries)
                
                if st.session_state.quality_scores:
                    avg_quality = sum(st.session_state.quality_scores) / len(st.session_state.quality_scores)
                    st.metric("Average Quality Score", f"{avg_quality:.2f}/10")
            
            # Clear conversation button
            if st.button("🗑️ Clear Conversation", type="secondary"):
                st.session_state.messages = []
                st.session_state.conversation_started = False
                st.session_state.total_queries = 0
                st.session_state.quality_scores = []
                st.rerun()
            
            # Security information
            st.markdown("### 🔒 Security Features")
            st.caption("""
            - Rate limiting enabled
            - Input sanitization active
            - Secure API handling
            - No data persistence
            """)
    
    def _render_chat_interface(self):
        """Render the main chat interface"""
        st.title("💬 Nawatech FAQ Assistant")
        st.markdown("Ask me anything about Nawatech services, products, or general information!")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show quality score for assistant messages
                if message["role"] == "assistant" and "quality_score" in message:
                    quality_score = message["quality_score"]
                    color = "green" if quality_score >= 7 else "orange" if quality_score >= 5 else "red"
                    st.caption(f"Quality Score: :{color}[{quality_score:.1f}/10]")
        
        # Welcome message
        if not st.session_state.conversation_started:
            with st.chat_message("assistant"):
                st.markdown("""
                👋 Hello! I'm your Nawatech FAQ assistant. I can help you with:
                
                - Information about Nawatech services
                - Company details and contact information
                - Technical support questions
                - Product inquiries
                
                What would you like to know?
                """)
            st.session_state.conversation_started = True
    
    def _handle_user_input(self):
        """Handle user input and generate responses"""
        if prompt := st.chat_input("Ask me about Nawatech..."):
            # Security check
            if not self.security_manager.is_request_allowed(st.session_state.user_id):
                st.error("⚠️ Rate limit exceeded. Please wait before sending another message.")
                return
            
            # Sanitize input
            sanitized_prompt = self.security_manager.sanitize_input(prompt)
            if not sanitized_prompt:
                st.error("⚠️ Invalid input detected. Please rephrase your question.")
                return
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": sanitized_prompt})
            st.session_state.total_queries += 1
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(sanitized_prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Get response from LLM handler
                        response_data = self.llm_handler.generate_response(
                            sanitized_prompt,
                            st.session_state.data_processor.vector_store
                        )
                        
                        response = response_data["answer"]
                        retrieved_docs = response_data.get("source_documents", [])
                        
                        # Calculate quality score
                        quality_score = self.quality_scorer.calculate_quality_score(
                            sanitized_prompt,
                            response,
                            retrieved_docs
                        )
                        
                        # Display response
                        st.markdown(response)
                        
                        # Display quality score
                        color = "green" if quality_score >= 7 else "orange" if quality_score >= 5 else "red"
                        st.caption(f"Quality Score: :{color}[{quality_score:.1f}/10]")
                        
                        # Show sources if available
                        if retrieved_docs:
                            with st.expander("📚 Source Information"):
                                for i, doc in enumerate(retrieved_docs[:3]):  # Show top 3 sources
                                    st.caption(f"**Source {i+1}:** {doc.page_content[:200]}...")
                        
                        # Store assistant message with quality score
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "quality_score": quality_score
                        })
                        
                        st.session_state.quality_scores.append(quality_score)
                        
                        logger.info(f"Generated response for user {st.session_state.user_id} with quality score {quality_score:.2f}")
                        
                    except Exception as e:
                        logger.error(f"Error generating response: {str(e)}")
                        error_message = "I apologize, but I'm experiencing technical difficulties. Please try again later."
                        st.error(error_message)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "quality_score": 0.0
                        })
    
    def _render_footer(self):
        """Render the footer with additional information"""
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏢 PT. Nawa Darsana Teknologi**")
            st.caption("Gedung Office 8, Lantai 18 Unit A, SCBD")
        
        with col2:
            st.markdown("**📞 Contact Information**")
            st.caption("Phone: 021-29552754")
        
        with col3:
            st.markdown("**⚡ Powered by**")
            st.caption("OpenAI GPT + LangChain + FAISS")
    
    def run(self):
        """Main application entry point"""
        try:
            # Check if API key is configured
            if not self.settings.OPENAI_API_KEY:
                st.error("⚠️ OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable.")
                st.stop()
            
            # Render UI components
            self._render_sidebar()
            self._render_chat_interface()
            self._handle_user_input()
            self._render_footer()
            
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            st.error(f"Application error: {str(e)}")


def main():
    """Main application function"""
    try:
        app = ChatbotApp()
        app.run()
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        st.error(f"Failed to start application: {str(e)}")


if __name__ == "__main__":
    main()