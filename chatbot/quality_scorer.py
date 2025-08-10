"""
Quality scorer for evaluating chatbot response quality
This module provides comprehensive scoring for answer relevance, completeness, and accuracy
"""

import logging
import re
from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document

logger = logging.getLogger(__name__)

class QualityScorer:
    """Evaluates the quality of chatbot responses"""
    
    def __init__(self):
        """Initialize the quality scorer"""
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.quality_weights = {
            'relevance': 0.3,
            'completeness': 0.25,
            'accuracy': 0.25,
            'clarity': 0.2
        }
        
        logger.info("Quality scorer initialized")
    
    def calculate_quality_score(self, query: str, response: str, source_documents: List[Document]) -> float:
        """
        Calculate overall quality score for a response
        
        Args:
            query: Original user query
            response: Generated response
            source_documents: Retrieved source documents
            
        Returns:
            Quality score between 0.0 and 10.0
        """
        try:
            # Calculate individual scores
            relevance_score = self._calculate_relevance_score(query, response, source_documents)
            completeness_score = self._calculate_completeness_score(response, source_documents)
            accuracy_score = self._calculate_accuracy_score(response, source_documents)
            clarity_score = self._calculate_clarity_score(response)
            
            # Weight and combine scores
            weighted_score = (
                relevance_score * self.quality_weights['relevance'] +
                completeness_score * self.quality_weights['completeness'] +
                accuracy_score * self.quality_weights['accuracy'] +
                clarity_score * self.quality_weights['clarity']
            )
            
            # Scale to 0-10
            final_score = weighted_score * 10
            
            logger.info(f"Quality score calculated: {final_score:.2f} (R:{relevance_score:.2f}, C:{completeness_score:.2f}, A:{accuracy_score:.2f}, Cl:{clarity_score:.2f})")
            
            return round(final_score, 1)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {str(e)}")
            return 5.0  # Default moderate score
    
    def _calculate_relevance_score(self, query: str, response: str, source_documents: List[Document]) -> float:
        """Calculate relevance score based on query-response similarity"""
        try:
            if not query or not response:
                return 0.0
            
            # Basic text similarity using TF-IDF
            texts = [query.lower(), response.lower()]
            
            # Add source document content for context
            for doc in source_documents[:3]:  # Use top 3 documents
                texts.append(doc.page_content.lower())
            
            if len(texts) < 2:
                return 0.5
            
            # Calculate TF-IDF vectors
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                
                # Calculate similarity between query and response
                query_response_similarity = cosine_similarity(
                    tfidf_matrix[0:1], tfidf_matrix[1:2]
                )[0][0]
                
                # If we have source documents, also check query-document similarity
                if len(source_documents) > 0 and len(texts) > 2:
                    query_doc_similarities = []
                    for i in range(2, min(len(texts), 5)):  # Check top 3 documents
                        similarity = cosine_similarity(
                            tfidf_matrix[0:1], tfidf_matrix[i:i+1]
                        )[0][0]
                        query_doc_similarities.append(similarity)
                    
                    # Average document relevance
                    avg_doc_relevance = np.mean(query_doc_similarities) if query_doc_similarities else 0.0
                    
                    # Combine query-response and query-document similarities
                    relevance_score = (query_response_similarity * 0.7 + avg_doc_relevance * 0.3)
                else:
                    relevance_score = query_response_similarity
                
                # Apply keyword boost
                keyword_boost = self._calculate_keyword_overlap(query, response)
                relevance_score = min(relevance_score + keyword_boost * 0.1, 1.0)
                
                return relevance_score
                
            except ValueError as e:
                # Handle case where TF-IDF fails (e.g., empty vocabulary)
                logger.warning(f"TF-IDF calculation failed, using fallback: {str(e)}")
                return self._calculate_keyword_overlap(query, response)
                
        except Exception as e:
            logger.error(f"Error calculating relevance score: {str(e)}")
            return 0.5
    
    def _calculate_completeness_score(self, response: str, source_documents: List[Document]) -> float:
        """Calculate completeness score based on response length and content coverage"""
        try:
            if not response:
                return 0.0
            
            # Response length factor (optimal range: 50-300 words)
            word_count = len(response.split())
            if word_count < 10:
                length_score = word_count / 10.0  # Penalty for very short responses
            elif word_count <= 50:
                length_score = 0.8 + (word_count - 10) / 40.0 * 0.2  # 0.8 to 1.0
            elif word_count <= 300:
                length_score = 1.0  # Optimal range
            else:
                length_score = max(0.7, 1.0 - (word_count - 300) / 500.0)  # Penalty for very long responses
            
            # Information coverage (based on source documents)
            coverage_score = 1.0
            if source_documents:
                # Check if response covers key information from sources
                source_content = " ".join([doc.page_content for doc in source_documents[:3]])
                coverage_score = self._calculate_information_coverage(response, source_content)
            
            # Structural completeness (has proper structure)
            structure_score = self._calculate_structure_score(response)
            
            # Combine scores
            completeness = (length_score * 0.4 + coverage_score * 0.4 + structure_score * 0.2)
            
            return completeness
            
        except Exception as e:
            logger.error(f"Error calculating completeness score: {str(e)}")
            return 0.5
    
    def _calculate_accuracy_score(self, response: str, source_documents: List[Document]) -> float:
        """Calculate accuracy score based on factual consistency with sources"""
        try:
            if not response or not source_documents:
                return 0.7  # Default score when no sources available
            
            # Extract key facts from response and sources
            response_facts = self._extract_key_facts(response)
            source_facts = []
            
            for doc in source_documents:
                doc_facts = self._extract_key_facts(doc.page_content)
                source_facts.extend(doc_facts)
            
            if not response_facts:
                return 0.5
            
            # Check fact consistency
            consistent_facts = 0
            total_facts = len(response_facts)
            
            for response_fact in response_facts:
                for source_fact in source_facts:
                    if self._facts_are_consistent(response_fact, source_fact):
                        consistent_facts += 1
                        break
            
            # Calculate accuracy ratio
            accuracy_ratio = consistent_facts / total_facts if total_facts > 0 else 0.5
            
            # Check for potential hallucinations (specific patterns)
            hallucination_penalty = self._detect_hallucinations(response, source_documents)
            accuracy_score = max(0.0, accuracy_ratio - hallucination_penalty)
            
            return accuracy_score
            
        except Exception as e:
            logger.error(f"Error calculating accuracy score: {str(e)}")
            return 0.6
    
    def _calculate_clarity_score(self, response: str) -> float:
        """Calculate clarity score based on readability and structure"""
        try:
            if not response:
                return 0.0
            
            # Sentence structure analysis
            sentences = re.split(r'[.!?]+', response)
            valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            
            if not valid_sentences:
                return 0.0
            
            # Average sentence length (optimal: 15-25 words)
            avg_sentence_length = np.mean([len(s.split()) for s in valid_sentences])
            if 15 <= avg_sentence_length <= 25:
                sentence_length_score = 1.0
            elif 10 <= avg_sentence_length <= 35:
                sentence_length_score = 0.8
            else:
                sentence_length_score = 0.6
            
            # Readability factors
            readability_score = self._calculate_readability_score(response)
            
            # Grammar and style (basic checks)
            grammar_score = self._calculate_grammar_score(response)
            
            # Professional tone check
            tone_score = self._calculate_tone_score(response)
            
            # Combine clarity components
            clarity = (
                sentence_length_score * 0.25 +
                readability_score * 0.25 +
                grammar_score * 0.25 +
                tone_score * 0.25
            )
            
            return clarity
            
        except Exception as e:
            logger.error(f"Error calculating clarity score: {str(e)}")
            return 0.7
    
    def _calculate_keyword_overlap(self, query: str, response: str) -> float:
        """Calculate keyword overlap between query and response"""
        try:
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            response_words = set(re.findall(r'\b\w+\b', response.lower()))
            
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            query_words -= stop_words
            response_words -= stop_words
            
            if not query_words:
                return 0.0
            
            overlap = len(query_words & response_words)
            return overlap / len(query_words)
            
        except Exception as e:
            logger.error(f"Error calculating keyword overlap: {str(e)}")
            return 0.0
    
    def _calculate_information_coverage(self, response: str, source_content: str) -> float:
        """Calculate how well the response covers information from sources"""
        try:
            if not source_content:
                return 1.0
            
            # Extract key phrases from both
            response_phrases = self._extract_key_phrases(response)
            source_phrases = self._extract_key_phrases(source_content)
            
            if not source_phrases:
                return 1.0
            
            # Calculate coverage
            covered_phrases = 0
            for source_phrase in source_phrases:
                for response_phrase in response_phrases:
                    if self._phrases_are_similar(source_phrase, response_phrase):
                        covered_phrases += 1
                        break
            
            coverage = covered_phrases / len(source_phrases)
            return min(coverage, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating information coverage: {str(e)}")
            return 0.7
    
    def _calculate_structure_score(self, response: str) -> float:
        """Calculate structural quality of the response"""
        try:
            structure_score = 0.0
            
            # Has proper sentences
            sentences = re.split(r'[.!?]+', response)
            if len([s for s in sentences if len(s.strip()) > 5]) >= 1:
                structure_score += 0.3
            
            # Has proper capitalization
            if response[0].isupper() if response else False:
                structure_score += 0.2
            
            # Ends with proper punctuation
            if response and response.strip()[-1] in '.!?':
                structure_score += 0.2
            
            # Not too many repetitive patterns
            words = response.lower().split()
            unique_words = len(set(words))
            total_words = len(words)
            if total_words > 0 and unique_words / total_words > 0.7:
                structure_score += 0.3
            
            return min(structure_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating structure score: {str(e)}")
            return 0.7
    
    def _extract_key_facts(self, text: str) -> List[str]:
        """Extract key factual statements from text"""
        try:
            # Simple fact extraction based on patterns
            facts = []
            
            # Look for factual patterns
            fact_patterns = [
                r'[A-Z][^.!?]*(?:is|are|was|were|has|have|will|can|provide|offer)[^.!?]*[.!?]',
                r'[A-Z][^.!?]*(?:located|based|founded|established)[^.!?]*[.!?]',
                r'[A-Z][^.!?]*(?:\d+|phone|email|address)[^.!?]*[.!?]',
            ]
            
            for pattern in fact_patterns:
                matches = re.findall(pattern, text)
                facts.extend(matches)
            
            # Clean and filter facts
            cleaned_facts = []
            for fact in facts:
                fact = fact.strip()
                if len(fact) > 10 and len(fact.split()) >= 3:
                    cleaned_facts.append(fact)
            
            return cleaned_facts[:5]  # Return top 5 facts
            
        except Exception as e:
            logger.error(f"Error extracting key facts: {str(e)}")
            return []
    
    def _facts_are_consistent(self, fact1: str, fact2: str) -> bool:
        """Check if two facts are consistent"""
        try:
            # Simple consistency check based on word overlap
            words1 = set(re.findall(r'\b\w+\b', fact1.lower()))
            words2 = set(re.findall(r'\b\w+\b', fact2.lower()))
            
            overlap = len(words1 & words2)
            return overlap >= 3  # At least 3 words in common
            
        except Exception as e:
            logger.error(f"Error checking fact consistency: {str(e)}")
            return False
    
    def _detect_hallucinations(self, response: str, source_documents: List[Document]) -> float:
        """Detect potential hallucinations in the response"""
        try:
            hallucination_penalty = 0.0
            
            # Check for specific information that should come from sources
            specific_patterns = [
                r'\b\d{4}-\d{7,10}\b',  # Phone numbers
                r'\b\d{1,5}\s+[A-Za-z\s]+(?:Street|Ave|Road|Blvd|Lane)\b',  # Addresses
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
            ]
            
            response_specific_info = []
            for pattern in specific_patterns:
                matches = re.findall(pattern, response)
                response_specific_info.extend(matches)
            
            if response_specific_info and source_documents:
                # Check if specific info is supported by sources
                source_text = " ".join([doc.page_content for doc in source_documents])
                for info in response_specific_info:
                    if info not in source_text:
                        hallucination_penalty += 0.1
            
            return min(hallucination_penalty, 0.5)  # Cap penalty at 0.5
            
        except Exception as e:
            logger.error(f"Error detecting hallucinations: {str(e)}")
            return 0.0
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate basic readability score"""
        try:
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            
            if not words or not sentences:
                return 0.5
            
            # Simple readability metrics
            avg_word_length = np.mean([len(word) for word in words])
            avg_sentence_length = len(words) / len(sentences)
            
            # Optimal ranges
            word_length_score = 1.0 if 4 <= avg_word_length <= 7 else 0.8
            sentence_length_score = 1.0 if 10 <= avg_sentence_length <= 25 else 0.8
            
            return (word_length_score + sentence_length_score) / 2
            
        except Exception as e:
            logger.error(f"Error calculating readability: {str(e)}")
            return 0.7
    
    def _calculate_grammar_score(self, text: str) -> float:
        """Calculate basic grammar score"""
        try:
            # Basic grammar checks
            grammar_score = 1.0
            
            # Check for basic capitalization
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and not sentence[0].isupper():
                    grammar_score -= 0.1
            
            # Check for repeated words
            words = text.lower().split()
            for i in range(len(words) - 1):
                if words[i] == words[i + 1]:
                    grammar_score -= 0.1
            
            return max(grammar_score, 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating grammar score: {str(e)}")
            return 0.8
    
    def _calculate_tone_score(self, text: str) -> float:
        """Calculate professional tone score"""
        try:
            tone_score = 1.0
            
            # Check for professional language
            professional_indicators = ['please', 'thank you', 'welcome', 'assist', 'help', 'information', 'service']
            unprofessional_indicators = ['yeah', 'nah', 'dunno', 'gonna', 'wanna']
            
            text_lower = text.lower()
            
            # Boost for professional language
            for indicator in professional_indicators:
                if indicator in text_lower:
                    tone_score = min(tone_score + 0.1, 1.0)
            
            # Penalty for unprofessional language
            for indicator in unprofessional_indicators:
                if indicator in text_lower:
                    tone_score -= 0.2
            
            return max(tone_score, 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating tone score: {str(e)}")
            return 0.8
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        try:
            # Simple phrase extraction (2-3 word combinations)
            words = re.findall(r'\b\w+\b', text.lower())
            phrases = []
            
            for i in range(len(words) - 1):
                if i < len(words) - 2:
                    phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
                phrases.append(f"{words[i]} {words[i+1]}")
            
            return list(set(phrases))[:10]  # Return unique phrases, limit to 10
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {str(e)}")
            return []
    
    def _phrases_are_similar(self, phrase1: str, phrase2: str) -> bool:
        """Check if two phrases are similar"""
        try:
            words1 = set(phrase1.split())
            words2 = set(phrase2.split())
            
            overlap = len(words1 & words2)
            return overlap >= len(words1) * 0.5  # At least 50% overlap
            
        except Exception as e:
            logger.error(f"Error checking phrase similarity: {str(e)}")
            return False
    
    def get_detailed_quality_report(self, query: str, response: str, source_documents: List[Document]) -> Dict:
        """
        Get detailed quality report
        
        Args:
            query: Original user query
            response: Generated response
            source_documents: Retrieved source documents
            
        Returns:
            Detailed quality report
        """
        try:
            relevance = self._calculate_relevance_score(query, response, source_documents)
            completeness = self._calculate_completeness_score(response, source_documents)
            accuracy = self._calculate_accuracy_score(response, source_documents)
            clarity = self._calculate_clarity_score(response)
            
            overall_score = (
                relevance * self.quality_weights['relevance'] +
                completeness * self.quality_weights['completeness'] +
                accuracy * self.quality_weights['accuracy'] +
                clarity * self.quality_weights['clarity']
            ) * 10
            
            return {
                'overall_score': round(overall_score, 1),
                'component_scores': {
                    'relevance': round(relevance * 10, 1),
                    'completeness': round(completeness * 10, 1),
                    'accuracy': round(accuracy * 10, 1),
                    'clarity': round(clarity * 10, 1)
                },
                'metrics': {
                    'word_count': len(response.split()),
                    'sentence_count': len(re.split(r'[.!?]+', response)),
                    'sources_used': len(source_documents)
                },
                'recommendations': self._generate_recommendations(relevance, completeness, accuracy, clarity)
            }
            
        except Exception as e:
            logger.error(f"Error generating detailed quality report: {str(e)}")
            return {
                'overall_score': 5.0,
                'error': str(e)
            }
    
    def _generate_recommendations(self, relevance: float, completeness: float, accuracy: float, clarity: float) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        if relevance < 0.7:
            recommendations.append("Improve response relevance to user query")
        
        if completeness < 0.7:
            recommendations.append("Provide more comprehensive information")
        
        if accuracy < 0.7:
            recommendations.append("Ensure factual accuracy and consistency with sources")
        
        if clarity < 0.7:
            recommendations.append("Improve response clarity and structure")
        
        if not recommendations:
            recommendations.append("Quality is good - maintain current standards")
        
        return recommendations