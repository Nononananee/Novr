"""
Enhanced Input Validation for Production Ready System
Provides comprehensive input validation, sanitization, and security checks.
"""

import re
import html
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field, validator, ValidationError
from enum import Enum
import unicodedata

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InputSanitizer:
    """Handles input sanitization and cleaning."""
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 100000) -> str:
        """
        Sanitize text input for security and quality.
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text truncated to {max_length} characters")
        
        # Remove null bytes and control characters
        text = text.replace('\0', '')
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\r\t')
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # HTML escape for safety
        text = html.escape(text, quote=False)
        
        return text.strip()
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename for filesystem safety.
        
        Args:
            filename: Input filename
            
        Returns:
            Safe filename
        """
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string")
        
        # Remove path separators and dangerous characters
        dangerous_chars = r'[<>:"/\\|?*\x00-\x1f]'
        filename = re.sub(dangerous_chars, '_', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:255-len(ext)-1] + '.' + ext if ext else name[:255]
        
        # Ensure not empty
        if not filename:
            filename = "untitled"
        
        return filename
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        if not isinstance(email, str):
            return False
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        if not isinstance(url, str):
            return False
        
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(url_pattern, url))


class ContentValidator:
    """Validates content for quality and appropriateness."""
    
    @staticmethod
    def validate_content_length(content: str, min_length: int = 1, max_length: int = 100000) -> Dict[str, Any]:
        """
        Validate content length.
        
        Args:
            content: Content to validate
            min_length: Minimum required length
            max_length: Maximum allowed length
            
        Returns:
            Validation result
        """
        if not isinstance(content, str):
            return {
                "valid": False,
                "error": "Content must be a string",
                "severity": ValidationSeverity.CRITICAL.value
            }
        
        content_length = len(content.strip())
        
        if content_length < min_length:
            return {
                "valid": False,
                "error": f"Content too short (minimum {min_length} characters)",
                "severity": ValidationSeverity.HIGH.value,
                "actual_length": content_length
            }
        
        if content_length > max_length:
            return {
                "valid": False,
                "error": f"Content too long (maximum {max_length} characters)",
                "severity": ValidationSeverity.HIGH.value,
                "actual_length": content_length
            }
        
        return {
            "valid": True,
            "length": content_length,
            "severity": ValidationSeverity.LOW.value
        }
    
    @staticmethod
    def validate_content_quality(content: str) -> Dict[str, Any]:
        """
        Validate content quality.
        
        Args:
            content: Content to validate
            
        Returns:
            Quality validation result
        """
        if not isinstance(content, str):
            return {
                "valid": False,
                "error": "Content must be a string",
                "quality_score": 0.0
            }
        
        content = content.strip()
        
        # Basic quality checks
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        
        # Check for reasonable word/sentence ratio
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        
        # Quality score calculation
        quality_score = 1.0
        issues = []
        
        # Too many repeated characters
        if re.search(r'(.)\1{10,}', content):
            quality_score -= 0.3
            issues.append("Excessive character repetition")
        
        # Too many repeated words
        words = content.lower().split()
        if len(words) > 10:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.3:  # More than 30% of content is one word
                quality_score -= 0.4
                issues.append("Excessive word repetition")
        
        # Very short sentences consistently
        if sentence_count > 3 and avg_words_per_sentence < 3:
            quality_score -= 0.2
            issues.append("Very short sentences")
        
        # Very long sentences consistently
        if sentence_count > 1 and avg_words_per_sentence > 50:
            quality_score -= 0.2
            issues.append("Very long sentences")
        
        quality_score = max(0.0, quality_score)
        
        return {
            "valid": quality_score >= 0.5,
            "quality_score": quality_score,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_words_per_sentence": avg_words_per_sentence,
            "issues": issues
        }
    
    @staticmethod
    def detect_potential_injection(content: str) -> Dict[str, Any]:
        """
        Detect potential injection attacks in content.
        
        Args:
            content: Content to check
            
        Returns:
            Injection detection result
        """
        if not isinstance(content, str):
            return {
                "safe": True,
                "threats_detected": [],
                "severity": ValidationSeverity.LOW.value
            }
        
        threats_detected = []
        severity = ValidationSeverity.LOW
        
        # SQL injection patterns
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC)\b)",
            r"(\b(UNION|OR|AND)\s+\d+\s*=\s*\d+)",
            r"('|\")?\s*;\s*(--|\#)",
            r"(\b(script|javascript|vbscript)\b)"
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                threats_detected.append("Potential SQL injection")
                severity = ValidationSeverity.HIGH
                break
        
        # XSS patterns
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>"
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                threats_detected.append("Potential XSS attack")
                severity = ValidationSeverity.HIGH
                break
        
        # Command injection patterns
        cmd_patterns = [
            r"(\||&&|;)\s*(rm|del|format|shutdown)",
            r"(cat|type)\s+(/etc/passwd|\.\.)",
            r"\$\((.*)\)",
            r"`[^`]+`"
        ]
        
        for pattern in cmd_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                threats_detected.append("Potential command injection")
                severity = ValidationSeverity.CRITICAL
                break
        
        return {
            "safe": len(threats_detected) == 0,
            "threats_detected": threats_detected,
            "severity": severity.value
        }


class EnhancedValidator:
    """Enhanced validation with multiple checks."""
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.content_validator = ContentValidator()
        self.validation_history = []
    
    def validate_input(self, data: Any, validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive input validation.
        
        Args:
            data: Data to validate
            validation_rules: Validation configuration
            
        Returns:
            Validation result
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "sanitized_data": data,
            "validation_id": len(self.validation_history)
        }
        
        try:
            # Type validation
            expected_type = validation_rules.get("type")
            if expected_type and not isinstance(data, expected_type):
                result["valid"] = False
                result["errors"].append(f"Expected {expected_type.__name__}, got {type(data).__name__}")
                return result
            
            # String validation
            if isinstance(data, str):
                result.update(self._validate_string(data, validation_rules))
            
            # Numeric validation
            elif isinstance(data, (int, float)):
                result.update(self._validate_numeric(data, validation_rules))
            
            # List validation
            elif isinstance(data, list):
                result.update(self._validate_list(data, validation_rules))
            
            # Dict validation
            elif isinstance(data, dict):
                result.update(self._validate_dict(data, validation_rules))
            
            # Record validation
            self.validation_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "sanitized_data": data,
                "validation_id": len(self.validation_history)
            }
    
    def _validate_string(self, data: str, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate string input."""
        result = {"errors": [], "warnings": []}
        
        # Length validation
        min_length = rules.get("min_length", 0)
        max_length = rules.get("max_length", 100000)
        
        length_result = self.content_validator.validate_content_length(data, min_length, max_length)
        if not length_result["valid"]:
            result["valid"] = False
            result["errors"].append(length_result["error"])
        
        # Sanitization
        if rules.get("sanitize", True):
            try:
                result["sanitized_data"] = self.sanitizer.sanitize_text(data, max_length)
            except Exception as e:
                result["errors"].append(f"Sanitization failed: {str(e)}")
        
        # Quality validation
        if rules.get("check_quality", True):
            quality_result = self.content_validator.validate_content_quality(data)
            if not quality_result["valid"]:
                result["warnings"].extend(quality_result.get("issues", []))
        
        # Security validation
        if rules.get("check_security", True):
            security_result = self.content_validator.detect_potential_injection(data)
            if not security_result["safe"]:
                result["valid"] = False
                result["errors"].extend(security_result["threats_detected"])
        
        return result
    
    def _validate_numeric(self, data: Union[int, float], rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate numeric input."""
        result = {"errors": [], "warnings": []}
        
        # Range validation
        min_value = rules.get("min_value")
        max_value = rules.get("max_value")
        
        if min_value is not None and data < min_value:
            result["valid"] = False
            result["errors"].append(f"Value {data} is below minimum {min_value}")
        
        if max_value is not None and data > max_value:
            result["valid"] = False
            result["errors"].append(f"Value {data} is above maximum {max_value}")
        
        # Integer validation
        if rules.get("integer_only", False) and isinstance(data, float) and not data.is_integer():
            result["valid"] = False
            result["errors"].append("Value must be an integer")
        
        return result
    
    def _validate_list(self, data: List[Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate list input."""
        result = {"errors": [], "warnings": [], "sanitized_data": []}
        
        # Length validation
        min_items = rules.get("min_items", 0)
        max_items = rules.get("max_items", 1000)
        
        if len(data) < min_items:
            result["valid"] = False
            result["errors"].append(f"List has {len(data)} items, minimum is {min_items}")
        
        if len(data) > max_items:
            result["valid"] = False
            result["errors"].append(f"List has {len(data)} items, maximum is {max_items}")
        
        # Item validation
        item_rules = rules.get("item_rules", {})
        if item_rules:
            for i, item in enumerate(data):
                item_result = self.validate_input(item, item_rules)
                if not item_result["valid"]:
                    result["valid"] = False
                    result["errors"].append(f"Item {i}: {', '.join(item_result['errors'])}")
                result["sanitized_data"].append(item_result["sanitized_data"])
        else:
            result["sanitized_data"] = data
        
        return result
    
    def _validate_dict(self, data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dictionary input."""
        result = {"errors": [], "warnings": [], "sanitized_data": {}}
        
        # Required fields validation
        required_fields = rules.get("required_fields", [])
        for field in required_fields:
            if field not in data:
                result["valid"] = False
                result["errors"].append(f"Required field '{field}' is missing")
        
        # Field validation
        field_rules = rules.get("field_rules", {})
        for field, value in data.items():
            if field in field_rules:
                field_result = self.validate_input(value, field_rules[field])
                if not field_result["valid"]:
                    result["valid"] = False
                    result["errors"].append(f"Field '{field}': {', '.join(field_result['errors'])}")
                result["sanitized_data"][field] = field_result["sanitized_data"]
            else:
                result["sanitized_data"][field] = value
        
        return result
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        if not self.validation_history:
            return {"total_validations": 0}
        
        total = len(self.validation_history)
        valid_count = sum(1 for v in self.validation_history if v["valid"])
        
        return {
            "total_validations": total,
            "valid_count": valid_count,
            "invalid_count": total - valid_count,
            "success_rate": valid_count / total if total > 0 else 0.0,
            "recent_validations": self.validation_history[-10:]
        }


# Global validator instance
enhanced_validator = EnhancedValidator()


# Convenience functions
def validate_text_input(text: str, 
                       min_length: int = 1, 
                       max_length: int = 100000,
                       sanitize: bool = True,
                       check_security: bool = True) -> Dict[str, Any]:
    """
    Validate text input with common rules.
    
    Args:
        text: Text to validate
        min_length: Minimum length
        max_length: Maximum length
        sanitize: Whether to sanitize
        check_security: Whether to check for security threats
        
    Returns:
        Validation result
    """
    rules = {
        "type": str,
        "min_length": min_length,
        "max_length": max_length,
        "sanitize": sanitize,
        "check_security": check_security
    }
    
    return enhanced_validator.validate_input(text, rules)


def validate_numeric_input(value: Union[int, float],
                          min_value: Optional[float] = None,
                          max_value: Optional[float] = None,
                          integer_only: bool = False) -> Dict[str, Any]:
    """
    Validate numeric input.
    
    Args:
        value: Numeric value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        integer_only: Whether to allow only integers
        
    Returns:
        Validation result
    """
    rules = {
        "type": (int, float),
        "min_value": min_value,
        "max_value": max_value,
        "integer_only": integer_only
    }
    
    return enhanced_validator.validate_input(value, rules)


def validate_list_input(data: List[Any],
                       min_items: int = 0,
                       max_items: int = 1000,
                       item_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate list input.
    
    Args:
        data: List to validate
        min_items: Minimum number of items
        max_items: Maximum number of items
        item_rules: Validation rules for each item
        
    Returns:
        Validation result
    """
    rules = {
        "type": list,
        "min_items": min_items,
        "max_items": max_items,
        "item_rules": item_rules or {}
    }
    
    return enhanced_validator.validate_input(data, rules)


# Convenience function for direct enhanced validation
def enhanced_validator_func(content: str, rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for enhanced validation.
    
    Args:
        content: Content to validate
        rules: Validation rules
        
    Returns:
        Validation result
    """
    return enhanced_validator.validate_input(content, rules)
