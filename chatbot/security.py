"""
Security manager for handling rate limiting, input sanitization, and security measures
This module protects against various attacks including prompt injection and brute force
"""

import logging
import re
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import hashlib
import html
import streamlit as st

logger = logging.getLogger(__name__)

class SecurityManager:
    """Handles security measures for the chatbot"""
    
    def __init__(self):
        """Initialize security manager"""
        self.rate_limit_storage = {}  # In production, use Redis or database
        self.max_requests_per_minute = 10
        self.max_requests_per_hour = 100
        self.blocked_patterns = self._load_blocked_patterns()
        self.suspicious_patterns = self._load_suspicious_patterns()
        
        # Initialize session-based tracking
        if 'security_violations' not in st.session_state:
            st.session_state.security_violations = 0
        
        logger.info("Security manager initialized")
    
    def _load_blocked_patterns(self) -> List[str]:
        """Load patterns that should be blocked"""
        return [
            # SQL Injection patterns
            r'(union\s+select|drop\s+table|delete\s+from|insert\s+into)',
            # XSS patterns
            r'(<script|javascript:|vbscript:|onload=|onerror=)',
            # Command injection
            r'(;|\||&&|\$\(|\`)',
            # Prompt injection attempts
            r'(ignore\s+previous|forget\s+instructions|system\s+prompt)',
            r'(act\s+as|pretend\s+to\s+be|roleplay)',
            r'(override\s+instructions|new\s+instructions)',
            # Data extraction attempts
            r'(show\s+me\s+your|what\s+are\s+your\s+instructions)',
            r'(reveal\s+your\s+prompt|display\s+system)',
        ]
    
    def _load_suspicious_patterns(self) -> List[str]:
        """Load patterns that are suspicious but not necessarily blocked"""
        return [
            # Repeated characters
            r'(.)\1{10,}',
            # Excessive special characters
            r'[!@#$%^&*()]{5,}',
            # Very long words
            r'\w{50,}',
            # Multiple question marks or exclamations
            r'[?!]{5,}',
        ]
    
    def is_request_allowed(self, user_id: str) -> bool:
        """
        Check if request is allowed based on rate limiting
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            True if request is allowed, False otherwise
        """
        try:
            current_time = datetime.now()
            
            # Initialize user tracking if not exists
            if user_id not in self.rate_limit_storage:
                self.rate_limit_storage[user_id] = {
                    'requests': [],
                    'violations': 0,
                    'last_violation': None
                }
            
            user_data = self.rate_limit_storage[user_id]
            
            # Clean old requests (older than 1 hour)
            user_data['requests'] = [
                req_time for req_time in user_data['requests']
                if current_time - req_time < timedelta(hours=1)
            ]
            
            # Check rate limits
            recent_requests = [
                req_time for req_time in user_data['requests']
                if current_time - req_time < timedelta(minutes=1)
            ]
            
            # Rate limit checks
            if len(recent_requests) >= self.max_requests_per_minute:
                self._log_violation(user_id, "Rate limit exceeded (per minute)")
                return False
            
            if len(user_data['requests']) >= self.max_requests_per_hour:
                self._log_violation(user_id, "Rate limit exceeded (per hour)")
                return False
            
            # Check for violation cooldown
            if user_data['last_violation']:
                cooldown_period = timedelta(minutes=5)
                if current_time - user_data['last_violation'] < cooldown_period:
                    return False
            
            # Record this request
            user_data['requests'].append(current_time)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in rate limiting check: {str(e)}")
            return True  # Allow request if there's an error
    
    def sanitize_input(self, user_input: str) -> Optional[str]:
        """
        Sanitize user input to prevent various attacks
        
        Args:
            user_input: Raw user input
            
        Returns:
            Sanitized input or None if input is blocked
        """
        try:
            if not user_input or not isinstance(user_input, str):
                return None
            
            # Basic length check
            if len(user_input) > 1000:
                logger.warning("Input too long, truncating")
                user_input = user_input[:1000]
            
            # Check for blocked patterns
            if self._contains_blocked_patterns(user_input):
                self._log_security_violation("Blocked pattern detected")
                return None
            
            # HTML escape to prevent XSS
            sanitized = html.escape(user_input)
            
            # Remove null bytes and control characters
            sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', sanitized)
            
            # Normalize whitespace
            sanitized = re.sub(r'\s+', ' ', sanitized).strip()
            
            # Check for suspicious patterns (warn but don't block)
            if self._contains_suspicious_patterns(sanitized):
                logger.warning(f"Suspicious pattern detected in input: {sanitized[:50]}...")
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Error sanitizing input: {str(e)}")
            return None
    
    def _contains_blocked_patterns(self, text: str) -> bool:
        """Check if text contains blocked patterns"""
        text_lower = text.lower()
        for pattern in self.blocked_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.warning(f"Blocked pattern found: {pattern}")
                return True
        return False
    
    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Check if text contains suspicious patterns"""
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _log_violation(self, user_id: str, violation_type: str):
        """Log security violation"""
        logger.warning(f"Security violation for user {user_id}: {violation_type}")
        
        if user_id in self.rate_limit_storage:
            self.rate_limit_storage[user_id]['violations'] += 1
            self.rate_limit_storage[user_id]['last_violation'] = datetime.now()
    
    def _log_security_violation(self, violation_type: str):
        """Log security violation in session state"""
        st.session_state.security_violations += 1
        logger.warning(f"Security violation: {violation_type}")
    
    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate API key format and basic security
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not api_key or not isinstance(api_key, str):
                return False
            
            # Basic format check for OpenAI API key
            if not api_key.startswith('sk-'):
                return False
            
            # Length check
            if len(api_key) < 40:
                return False
            
            # Character set check
            if not re.match(r'^sk-[A-Za-z0-9]+$', api_key):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating API key: {str(e)}")
            return False
    
    def hash_user_id(self, user_identifier: str) -> str:
        """
        Create a hashed user ID for privacy
        
        Args:
            user_identifier: Original user identifier
            
        Returns:
            Hashed user ID
        """
        try:
            # Add salt for security
            salt = "nawatech_chatbot_salt_2024"
            combined = f"{user_identifier}{salt}"
            
            # Create hash
            hashed = hashlib.sha256(combined.encode()).hexdigest()
            
            return f"user_{hashed[:16]}"
            
        except Exception as e:
            logger.error(f"Error hashing user ID: {str(e)}")
            return f"user_{int(time.time())}"
    
    def detect_prompt_injection(self, user_input: str) -> Dict[str, any]:
        """
        Advanced prompt injection detection
        
        Args:
            user_input: User input to analyze
            
        Returns:
            Detection results
        """
        try:
            detection_result = {
                'is_injection': False,
                'confidence': 0.0,
                'detected_patterns': [],
                'risk_level': 'low'
            }
            
            input_lower = user_input.lower()
            
            # High-risk injection patterns
            high_risk_patterns = [
                (r'ignore\s+(all\s+)?previous\s+instructions', 0.9),
                (r'forget\s+(everything|all)\s+(you\s+)?(know|learned)', 0.9),
                (r'you\s+are\s+now\s+a\s+different', 0.8),
                (r'system\s*:\s*you\s+are', 0.8),
                (r'new\s+instruction(s)?:\s*you', 0.8),
                (r'override\s+your\s+programming', 0.9),
                (r'act\s+as\s+if\s+you\s+are\s+not', 0.7),
                (r'pretend\s+you\s+are\s+a\s+different', 0.7),
            ]
            
            # Medium-risk patterns
            medium_risk_patterns = [
                (r'what\s+are\s+your\s+instructions', 0.5),
                (r'show\s+me\s+your\s+prompt', 0.6),
                (r'reveal\s+your\s+system\s+message', 0.6),
                (r'tell\s+me\s+about\s+your\s+training', 0.4),
            ]
            
            total_confidence = 0.0
            detected_patterns = []
            
            # Check high-risk patterns
            for pattern, confidence in high_risk_patterns:
                if re.search(pattern, input_lower):
                    detected_patterns.append(pattern)
                    total_confidence += confidence
            
            # Check medium-risk patterns
            for pattern, confidence in medium_risk_patterns:
                if re.search(pattern, input_lower):
                    detected_patterns.append(pattern)
                    total_confidence += confidence
            
            # Normalize confidence
            final_confidence = min(total_confidence, 1.0)
            
            # Determine risk level and injection status
            if final_confidence >= 0.7:
                detection_result.update({
                    'is_injection': True,
                    'risk_level': 'high'
                })
            elif final_confidence >= 0.4:
                detection_result.update({
                    'is_injection': True,
                    'risk_level': 'medium'
                })
            elif final_confidence >= 0.2:
                detection_result.update({
                    'risk_level': 'low'
                })
            
            detection_result.update({
                'confidence': final_confidence,
                'detected_patterns': detected_patterns
            })
            
            if detection_result['is_injection']:
                logger.warning(f"Prompt injection detected: confidence={final_confidence:.2f}, patterns={detected_patterns}")
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Error in prompt injection detection: {str(e)}")
            return {
                'is_injection': False,
                'confidence': 0.0,
                'detected_patterns': [],
                'risk_level': 'unknown'
            }
    
    def get_security_report(self, user_id: str) -> Dict[str, any]:
        """
        Get security report for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Security report
        """
        try:
            report = {
                'user_id': user_id,
                'total_requests': 0,
                'violations': 0,
                'last_violation': None,
                'current_rate_limit_status': 'normal',
                'session_violations': st.session_state.get('security_violations', 0)
            }
            
            if user_id in self.rate_limit_storage:
                user_data = self.rate_limit_storage[user_id]
                report.update({
                    'total_requests': len(user_data['requests']),
                    'violations': user_data['violations'],
                    'last_violation': user_data['last_violation']
                })
                
                # Check current rate limit status
                current_time = datetime.now()
                recent_requests = [
                    req_time for req_time in user_data['requests']
                    if current_time - req_time < timedelta(minutes=1)
                ]
                
                if len(recent_requests) >= self.max_requests_per_minute * 0.8:
                    report['current_rate_limit_status'] = 'warning'
                elif len(recent_requests) >= self.max_requests_per_minute:
                    report['current_rate_limit_status'] = 'limited'
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating security report: {str(e)}")
            return {
                'user_id': user_id,
                'error': str(e)
            }