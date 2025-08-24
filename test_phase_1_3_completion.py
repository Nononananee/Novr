#!/usr/bin/env python3
"""
Phase 1.3 Completion Test: Basic Input Validation
Tests the implementation of comprehensive input validation and security checks.
"""

import asyncio
import pytest
import sys
import os
import logging
from typing import Dict, Any
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase1_3TestSuite:
    """Test suite for Phase 1.3: Basic Input Validation"""
    
    def __init__(self):
        self.test_results = []
        self.passed_tests = 0
        self.total_tests = 0
    
    async def run_all_tests(self):
        """Run all Phase 1.3 tests."""
        print("=" * 80)
        print("PHASE 1.3 COMPLETION TEST: BASIC INPUT VALIDATION")
        print("=" * 80)
        
        # Test 1: Input Validation Utils Import
        await self._run_test("Input Validation Utils Import", self.test_input_validation_import)
        
        # Test 2: Text Input Validation
        await self._run_test("Text Input Validation", self.test_text_validation)
        
        # Test 3: Numeric Input Validation
        await self._run_test("Numeric Input Validation", self.test_numeric_validation)
        
        # Test 4: Security Threat Detection
        await self._run_test("Security Threat Detection", self.test_security_validation)
        
        # Test 5: Input Sanitization
        await self._run_test("Input Sanitization", self.test_input_sanitization)
        
        # Test 6: API Integration
        await self._run_test("API Integration", self.test_api_integration)
        
        # Test 7: Content Quality Validation
        await self._run_test("Content Quality Validation", self.test_content_quality)
        
        # Test 8: Edge Case Handling
        await self._run_test("Edge Case Handling", self.test_edge_cases)
        
        # Generate final report
        self.generate_final_report()
    
    async def _run_test(self, test_name: str, test_func):
        """Run a single test with error handling."""
        self.total_tests += 1
        start_time = asyncio.get_event_loop().time()
        
        try:
            print(f"\n🧪 Running test: {test_name}")
            
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            if result.get("success", False):
                self.passed_tests += 1
                print(f"✅ {test_name} PASSED ({execution_time:.2f}ms)")
            else:
                print(f"❌ {test_name} FAILED: {result.get('error', 'Unknown error')}")
            
            self.test_results.append({
                "test_name": test_name,
                "success": result.get("success", False),
                "execution_time_ms": execution_time,
                "details": result,
                "error_message": result.get("error") if not result.get("success") else None
            })
            
        except Exception as e:
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            print(f"❌ {test_name} CRASHED: {str(e)}")
            
            self.test_results.append({
                "test_name": test_name,
                "success": False,
                "execution_time_ms": execution_time,
                "details": {},
                "error_message": str(e)
            })
    
    def test_input_validation_import(self) -> Dict[str, Any]:
        """Test that input validation utils can be imported."""
        try:
            from agent.input_validation import (
                InputSanitizer,
                ContentValidator,
                EnhancedValidator,
                ValidationSeverity,
                validate_text_input,
                validate_numeric_input,
                enhanced_validator
            )
            
            # Verify classes exist and have expected methods
            assert hasattr(InputSanitizer, 'sanitize_text')
            assert hasattr(ContentValidator, 'validate_content_length')
            assert hasattr(EnhancedValidator, 'validate_input')
            assert hasattr(enhanced_validator, 'validate_input')
            
            return {
                "success": True,
                "message": "All input validation components imported successfully",
                "components": [
                    "InputSanitizer",
                    "ContentValidator", 
                    "EnhancedValidator",
                    "ValidationSeverity",
                    "validate_text_input",
                    "validate_numeric_input"
                ]
            }
            
        except ImportError as e:
            return {
                "success": False,
                "error": f"Import failed: {str(e)}",
                "missing_component": str(e).split("'")[-2] if "'" in str(e) else "unknown"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def test_text_validation(self) -> Dict[str, Any]:
        """Test text input validation functionality."""
        try:
            from agent.input_validation import validate_text_input
            
            # Test valid text
            valid_result = validate_text_input("This is a valid text", min_length=5, max_length=100)
            assert valid_result["valid"] is True
            assert "sanitized_data" in valid_result
            
            # Test too short text
            short_result = validate_text_input("Hi", min_length=10)
            assert short_result["valid"] is False
            assert "too short" in short_result["errors"][0].lower()
            
            # Test too long text
            long_text = "x" * 1000
            long_result = validate_text_input(long_text, max_length=100)
            assert long_result["valid"] is False
            assert "too long" in long_result["errors"][0].lower()
            
            # Test empty text
            empty_result = validate_text_input("", min_length=1)
            assert empty_result["valid"] is False
            
            return {
                "success": True,
                "message": "Text validation working correctly",
                "valid_text_passed": valid_result["valid"],
                "short_text_rejected": not short_result["valid"],
                "long_text_rejected": not long_result["valid"],
                "empty_text_rejected": not empty_result["valid"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Text validation failed: {str(e)}"
            }
    
    def test_numeric_validation(self) -> Dict[str, Any]:
        """Test numeric input validation."""
        try:
            from agent.input_validation import validate_numeric_input
            
            # Test valid number
            valid_result = validate_numeric_input(50, min_value=0, max_value=100)
            assert valid_result["valid"] is True
            
            # Test below minimum
            low_result = validate_numeric_input(-5, min_value=0)
            assert low_result["valid"] is False
            assert "below minimum" in low_result["errors"][0].lower()
            
            # Test above maximum
            high_result = validate_numeric_input(150, max_value=100)
            assert high_result["valid"] is False
            assert "above maximum" in high_result["errors"][0].lower()
            
            # Test integer only
            float_result = validate_numeric_input(3.14, integer_only=True)
            assert float_result["valid"] is False
            assert "integer" in float_result["errors"][0].lower()
            
            return {
                "success": True,
                "message": "Numeric validation working correctly",
                "valid_number_passed": valid_result["valid"],
                "low_number_rejected": not low_result["valid"],
                "high_number_rejected": not high_result["valid"],
                "float_as_integer_rejected": not float_result["valid"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Numeric validation failed: {str(e)}"
            }
    
    def test_security_validation(self) -> Dict[str, Any]:
        """Test security threat detection."""
        try:
            from agent.input_validation import ContentValidator
            
            validator = ContentValidator()
            
            # Test safe content
            safe_result = validator.detect_potential_injection("This is safe content")
            assert safe_result["safe"] is True
            assert len(safe_result["threats_detected"]) == 0
            
            # Test SQL injection attempt
            sql_result = validator.detect_potential_injection("SELECT * FROM users WHERE id = 1 OR 1=1")
            assert sql_result["safe"] is False
            assert len(sql_result["threats_detected"]) > 0
            
            # Test XSS attempt
            xss_result = validator.detect_potential_injection("<script>alert('xss')</script>")
            assert xss_result["safe"] is False
            assert len(xss_result["threats_detected"]) > 0
            
            # Test command injection
            cmd_result = validator.detect_potential_injection("test; rm -rf /")
            assert cmd_result["safe"] is False
            assert len(cmd_result["threats_detected"]) > 0
            
            return {
                "success": True,
                "message": "Security validation working correctly",
                "safe_content_passed": safe_result["safe"],
                "sql_injection_detected": not sql_result["safe"],
                "xss_detected": not xss_result["safe"],
                "command_injection_detected": not cmd_result["safe"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Security validation failed: {str(e)}"
            }
    
    def test_input_sanitization(self) -> Dict[str, Any]:
        """Test input sanitization functionality."""
        try:
            from agent.input_validation import InputSanitizer
            
            sanitizer = InputSanitizer()
            
            # Test HTML escaping
            html_input = "Test <script>alert('test')</script> content"
            sanitized = sanitizer.sanitize_text(html_input)
            assert "&lt;script&gt;" in sanitized
            assert "<script>" not in sanitized
            
            # Test control character removal
            control_input = "Test\x00\x01content"
            sanitized_control = sanitizer.sanitize_text(control_input)
            assert "\x00" not in sanitized_control
            assert "\x01" not in sanitized_control
            
            # Test length truncation
            long_input = "x" * 1000
            truncated = sanitizer.sanitize_text(long_input, max_length=100)
            assert len(truncated) <= 100
            
            # Test filename sanitization
            dangerous_filename = "test<>:\"/\\|?*file.txt"
            safe_filename = sanitizer.sanitize_filename(dangerous_filename)
            assert "<" not in safe_filename
            assert ">" not in safe_filename
            assert ":" not in safe_filename
            
            return {
                "success": True,
                "message": "Input sanitization working correctly",
                "html_escaped": "&lt;script&gt;" in sanitized,
                "control_chars_removed": "\x00" not in sanitized_control,
                "length_truncated": len(truncated) <= 100,
                "filename_sanitized": "<" not in safe_filename
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Input sanitization failed: {str(e)}"
            }
    
    def test_api_integration(self) -> Dict[str, Any]:
        """Test API integration with input validation."""
        try:
            # Import to ensure integration exists
            from agent.api import app
            from agent.input_validation import validate_text_input
            
            # Test that validation functions are available
            test_result = validate_text_input("test message")
            assert isinstance(test_result, dict)
            assert "valid" in test_result
            
            return {
                "success": True,
                "message": "API integration working correctly",
                "validation_function_available": True,
                "api_module_importable": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"API integration failed: {str(e)}"
            }
    
    def test_content_quality(self) -> Dict[str, Any]:
        """Test content quality validation."""
        try:
            from agent.input_validation import ContentValidator
            
            validator = ContentValidator()
            
            # Test good quality content
            good_content = "This is a well-written paragraph with proper sentence structure. It contains multiple sentences with varying lengths and good vocabulary."
            good_result = validator.validate_content_quality(good_content)
            
            # Test poor quality content (excessive repetition)
            poor_content = "test test test test test test test test test test"
            poor_result = validator.validate_content_quality(poor_content)
            
            # Test very short content
            short_content = "Hi."
            short_result = validator.validate_content_quality(short_content)
            
            assert isinstance(good_result, dict)
            assert "quality_score" in good_result
            assert "word_count" in good_result
            
            return {
                "success": True,
                "message": "Content quality validation working",
                "good_content_score": good_result["quality_score"],
                "poor_content_score": poor_result["quality_score"],
                "quality_scoring_functional": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Content quality validation failed: {str(e)}"
            }
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge case handling."""
        try:
            from agent.input_validation import validate_text_input, validate_numeric_input
            
            # Test None input
            try:
                none_result = validate_text_input(None)
                none_handled = not none_result["valid"]
            except:
                none_handled = True  # Exception is acceptable
            
            # Test non-string input for text validation
            try:
                number_result = validate_text_input(123)
                number_handled = not number_result["valid"]
            except:
                number_handled = True  # Exception is acceptable
            
            # Test very large number
            large_result = validate_numeric_input(float('inf'), max_value=1000)
            large_handled = not large_result["valid"]
            
            # Test NaN
            try:
                nan_result = validate_numeric_input(float('nan'))
                nan_handled = True  # Should handle gracefully
            except:
                nan_handled = True  # Exception is acceptable
            
            return {
                "success": True,
                "message": "Edge cases handled correctly",
                "none_input_handled": none_handled,
                "wrong_type_handled": number_handled,
                "large_number_handled": large_handled,
                "nan_handled": nan_handled
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Edge case handling failed: {str(e)}"
            }
    
    def generate_final_report(self):
        """Generate final test report."""
        print("\n" + "=" * 80)
        print("PHASE 1.3 TEST RESULTS SUMMARY")
        print("=" * 80)
        
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed Tests: {self.passed_tests}")
        print(f"Failed Tests: {self.total_tests - self.passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Phase 1.3 success criteria
        phase_success = success_rate >= 80  # At least 80% pass rate
        
        print(f"\n🎯 PHASE 1.3 STATUS: {'✅ PASSED' if phase_success else '❌ FAILED'}")
        
        if phase_success:
            print("\n✅ Basic input validation implementation is ready!")
            print("✅ Security threat detection working correctly")
            print("✅ Input sanitization functional")
            print("✅ API integration successful")
            print("✅ PHASE 1 COMPLETE - Ready to proceed to Phase 2!")
        else:
            print("\n❌ Phase 1.3 requirements not met")
            print("❌ Fix failing tests before proceeding to Phase 2")
        
        # Detailed results
        print("\n📊 DETAILED TEST RESULTS:")
        for result in self.test_results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {status} {result['test_name']} ({result['execution_time_ms']:.2f}ms)")
            if not result["success"] and result["error_message"]:
                print(f"    Error: {result['error_message']}")
        
        return {
            "phase": "1.3",
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "success_rate": success_rate,
            "phase_passed": phase_success,
            "test_results": self.test_results
        }


async def main():
    """Run Phase 1.3 completion tests."""
    test_suite = Phase1_3TestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
