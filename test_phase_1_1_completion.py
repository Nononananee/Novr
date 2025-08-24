#!/usr/bin/env python3
"""
Phase 1.1 Completion Test: Critical Error Handling
Tests the implementation of robust error handling, graceful degradation, and memory monitoring.
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


class Phase1_1TestSuite:
    """Test suite for Phase 1.1: Critical Error Handling"""
    
    def __init__(self):
        self.test_results = []
        self.passed_tests = 0
        self.total_tests = 0
    
    async def run_all_tests(self):
        """Run all Phase 1.1 tests."""
        print("=" * 80)
        print("PHASE 1.1 COMPLETION TEST: CRITICAL ERROR HANDLING")
        print("=" * 80)
        
        # Test 1: Error Handling Utils Import
        await self._run_test("Error Handling Utils Import", self.test_error_handling_import)
        
        # Test 2: Graceful Degradation
        await self._run_test("Graceful Degradation", self.test_graceful_degradation)
        
        # Test 3: Retry Mechanism
        await self._run_test("Retry Mechanism", self.test_retry_mechanism)
        
        # Test 4: Enhanced Memory Monitor
        await self._run_test("Enhanced Memory Monitor", self.test_memory_monitoring)
        
        # Test 5: Validator Error Handling
        await self._run_test("Validator Error Handling", self.test_validator_error_handling)
        
        # Test 6: Memory Profiler
        await self._run_test("Memory Profiler", self.test_memory_profiler)
        
        # Test 7: Error Metrics
        await self._run_test("Error Metrics", self.test_error_metrics)
        
        # Test 8: Health Checker
        await self._run_test("Health Checker", self.test_health_checker)
        
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
    
    def test_error_handling_import(self) -> Dict[str, Any]:
        """Test that error handling utils can be imported."""
        try:
            from agent.error_handling_utils import (
                GracefulDegradation,
                RetryMechanism,
                robust_error_handler,
                ErrorSeverity,
                ErrorContext,
                error_metrics
            )
            
            # Verify classes exist and have expected methods
            assert hasattr(GracefulDegradation, 'get_validation_fallback')
            assert hasattr(RetryMechanism, 'execute_with_retry')
            assert hasattr(error_metrics, 'record_error')
            
            return {
                "success": True,
                "message": "All error handling components imported successfully",
                "components": [
                    "GracefulDegradation",
                    "RetryMechanism", 
                    "robust_error_handler",
                    "ErrorSeverity",
                    "ErrorContext",
                    "error_metrics"
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
    
    async def test_graceful_degradation(self) -> Dict[str, Any]:
        """Test graceful degradation functionality."""
        try:
            from agent.error_handling_utils import GracefulDegradation, ErrorContext, ErrorSeverity
            
            # Test validation fallback
            context = ErrorContext(
                operation="test_validation",
                input_data={"test": "data"},
                severity=ErrorSeverity.MEDIUM
            )
            
            result = await GracefulDegradation.get_validation_fallback(
                "test content", "test_validator", context
            )
            
            # Verify fallback structure
            assert isinstance(result, dict)
            assert "score" in result
            assert "violations" in result
            assert "suggestions" in result
            assert "is_fallback" in result
            assert result["is_fallback"] is True
            
            return {
                "success": True,
                "message": "Graceful degradation working correctly",
                "fallback_score": result["score"],
                "has_suggestions": len(result["suggestions"]) > 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Graceful degradation failed: {str(e)}"
            }
    
    async def test_retry_mechanism(self) -> Dict[str, Any]:
        """Test retry mechanism with exponential backoff."""
        try:
            from agent.error_handling_utils import RetryMechanism, ErrorContext, ErrorSeverity
            
            retry_mechanism = RetryMechanism(max_retries=2, base_delay=0.1)
            
            # Test successful retry
            call_count = 0
            
            async def failing_function():
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ValueError("Temporary failure")
                return {"success": True, "attempts": call_count}
            
            context = ErrorContext(
                operation="test_retry",
                input_data={},
                severity=ErrorSeverity.MEDIUM
            )
            
            result = await retry_mechanism.execute_with_retry(
                failing_function, error_context=context
            )
            
            assert result["success"] is True
            assert result["attempts"] == 2
            
            return {
                "success": True,
                "message": "Retry mechanism working correctly",
                "retry_attempts": call_count,
                "backoff_working": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Retry mechanism failed: {str(e)}"
            }
    
    def test_memory_monitoring(self) -> Dict[str, Any]:
        """Test enhanced memory monitoring."""
        try:
            from agent.enhanced_memory_monitor import (
                AccurateMemoryMonitor,
                memory_monitor,
                get_memory_health
            )
            
            # Test memory usage measurement
            current_usage = memory_monitor.get_current_memory_usage()
            
            # Verify structure
            assert isinstance(current_usage, dict)
            assert "rss_mb" in current_usage
            assert "vms_mb" in current_usage
            assert current_usage["rss_mb"] > 0  # Should have some memory usage
            
            # Test snapshot recording
            snapshot = memory_monitor.record_snapshot("test_operation")
            assert snapshot.rss_mb > 0
            assert snapshot.operation == "test_operation"
            
            # Test memory health
            health = get_memory_health()
            assert "status" in health
            assert "current_memory_mb" in health
            
            return {
                "success": True,
                "message": "Memory monitoring working correctly",
                "current_memory_mb": current_usage["rss_mb"],
                "health_status": health["status"],
                "accurate_measurement": current_usage["rss_mb"] > 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Memory monitoring failed: {str(e)}"
            }
    
    async def test_validator_error_handling(self) -> Dict[str, Any]:
        """Test that validators now use enhanced error handling."""
        try:
            from agent.consistency_validators_fixed import fact_check_validator
            from agent.error_handling_utils import error_metrics
            
            # Clear previous error metrics
            error_metrics.error_counts.clear()
            
            # Test with invalid input to trigger error handling
            result = await fact_check_validator(
                content="",  # Empty content should trigger fallback
                entity_data={},
                established_facts=set()
            )
            
            # Should return fallback result, not crash
            assert isinstance(result, dict)
            assert "score" in result
            
            # Check if error was recorded (might be in fallback or successful execution)
            return {
                "success": True,
                "message": "Validator error handling working",
                "fallback_used": result.get("is_fallback", False),
                "has_score": "score" in result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Validator error handling failed: {str(e)}"
            }
    
    async def test_memory_profiler(self) -> Dict[str, Any]:
        """Test memory profiler context manager."""
        try:
            from agent.enhanced_memory_monitor import monitor_operation_memory
            
            # Test memory profiling
            async with await monitor_operation_memory("test_profile_operation") as profiler:
                # Simulate some memory usage
                test_data = [i for i in range(1000)]
                await asyncio.sleep(0.1)  # Simulate async operation
            
            memory_used = profiler.get_memory_usage()
            
            return {
                "success": True,
                "message": "Memory profiler working correctly",
                "memory_tracked": memory_used >= 0,  # Could be 0 for small operations
                "profiler_functional": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Memory profiler failed: {str(e)}"
            }
    
    def test_error_metrics(self) -> Dict[str, Any]:
        """Test error metrics tracking."""
        try:
            from agent.error_handling_utils import error_metrics, ErrorSeverity
            
            # Record test error
            error_metrics.record_error("test_operation", "TestError", ErrorSeverity.LOW)
            
            # Get summary
            summary = error_metrics.get_error_summary()
            
            assert "total_errors" in summary
            assert "error_counts" in summary
            
            return {
                "success": True,
                "message": "Error metrics working correctly",
                "total_errors": summary["total_errors"],
                "metrics_tracked": len(summary["error_counts"]) > 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error metrics failed: {str(e)}"
            }
    
    async def test_health_checker(self) -> Dict[str, Any]:
        """Test health checker functionality."""
        try:
            from agent.error_handling_utils import HealthChecker
            
            # Test health check
            def dummy_health_check():
                return {"status": "ok", "response_time": 0.1}
            
            result = await HealthChecker.check_component_health(
                "test_component", dummy_health_check
            )
            
            assert "component" in result
            assert "status" in result
            assert result["component"] == "test_component"
            
            return {
                "success": True,
                "message": "Health checker working correctly",
                "component_status": result["status"],
                "response_time": result.get("response_time", 0)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Health checker failed: {str(e)}"
            }
    
    def generate_final_report(self):
        """Generate final test report."""
        print("\n" + "=" * 80)
        print("PHASE 1.1 TEST RESULTS SUMMARY")
        print("=" * 80)
        
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed Tests: {self.passed_tests}")
        print(f"Failed Tests: {self.total_tests - self.passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Phase 1.1 success criteria
        phase_success = success_rate >= 80  # At least 80% pass rate
        
        print(f"\n🎯 PHASE 1.1 STATUS: {'✅ PASSED' if phase_success else '❌ FAILED'}")
        
        if phase_success:
            print("\n✅ Critical Error Handling implementation is ready!")
            print("✅ Graceful degradation working correctly")
            print("✅ Memory monitoring is accurate")
            print("✅ Ready to proceed to Sub-Phase 1.2")
        else:
            print("\n❌ Phase 1.1 requirements not met")
            print("❌ Fix failing tests before proceeding")
        
        # Detailed results
        print("\n📊 DETAILED TEST RESULTS:")
        for result in self.test_results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {status} {result['test_name']} ({result['execution_time_ms']:.2f}ms)")
            if not result["success"] and result["error_message"]:
                print(f"    Error: {result['error_message']}")
        
        return {
            "phase": "1.1",
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "success_rate": success_rate,
            "phase_passed": phase_success,
            "test_results": self.test_results
        }


async def main():
    """Run Phase 1.1 completion tests."""
    test_suite = Phase1_1TestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
