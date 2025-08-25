"""Test framework verification - simple tests to verify testing infrastructure."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio


@pytest.mark.critical
class TestFrameworkVerification:
    """Verify that the testing framework is working correctly."""
    
    def test_basic_test_execution(self):
        """Test that basic test execution works."""
        assert True, "Basic test execution should work"
    
    def test_mock_functionality(self):
        """Test that mocking functionality works."""
        mock_obj = Mock()
        mock_obj.test_method.return_value = "mocked_result"
        
        result = mock_obj.test_method()
        assert result == "mocked_result"
        mock_obj.test_method.assert_called_once()
    
    def test_patch_functionality(self):
        """Test that patching functionality works."""
        # Use a safer function to patch that won't cause recursion
        import os
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            result = os.path.exists('/fake/path')
            assert result is True
            mock_exists.assert_called_once_with('/fake/path')
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test that async test functionality works."""
        async def async_function():
            await asyncio.sleep(0.001)
            return "async_result"
        
        result = await async_function()
        assert result == "async_result"
    
    @pytest.mark.asyncio
    async def test_async_mock_functionality(self):
        """Test that async mocking functionality works."""
        mock_async = AsyncMock()
        mock_async.return_value = "async_mock_result"
        
        result = await mock_async()
        assert result == "async_mock_result"
        mock_async.assert_called_once()


@pytest.mark.medium
class TestPriorityMarkers:
    """Test that priority markers work correctly."""
    
    def test_medium_priority_marker(self):
        """Test medium priority marker."""
        assert True, "Medium priority test should work"


@pytest.mark.unit
class TestUnitMarkers:
    """Test that unit test markers work correctly."""
    
    def test_unit_marker(self):
        """Test unit marker."""
        assert True, "Unit test marker should work"


@pytest.mark.database
class TestDatabaseMarkers:
    """Test that database markers work correctly."""
    
    def test_database_marker(self):
        """Test database marker."""
        assert True, "Database test marker should work"


class TestBasicFunctionality:
    """Test basic Python functionality that our tests depend on."""
    
    def test_json_handling(self):
        """Test JSON handling works."""
        import json
        
        data = {"test": "value", "number": 42}
        json_str = json.dumps(data)
        parsed_data = json.loads(json_str)
        
        assert parsed_data == data
    
    def test_datetime_handling(self):
        """Test datetime handling works."""
        from datetime import datetime
        
        now = datetime.now()
        iso_string = now.isoformat()
        
        assert isinstance(iso_string, str)
        assert "T" in iso_string
    
    def test_exception_handling(self):
        """Test exception handling works."""
        with pytest.raises(ValueError):
            raise ValueError("Test exception")
    
    def test_parametrized_test(self):
        """Test parametrized tests work."""
        test_cases = [
            (1, 2, 3),
            (5, 5, 10),
            (0, 100, 100)
        ]
        
        for a, b, expected in test_cases:
            assert a + b == expected