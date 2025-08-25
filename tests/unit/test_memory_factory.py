"""
Comprehensive unit tests for memory.memory_factory module.
Tests memory system factory functions and component validation.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import logging

from memory.memory_factory import (
    create_integrated_memory_system,
    create_emotional_memory_system,
    validate_memory_system_components
)
from memory.integrated_memory_system import IntegratedNovelMemorySystem
from memory.emotional_memory_system import EmotionalMemorySystem


class TestCreateIntegratedMemorySystem:
    """Test create_integrated_memory_system factory function."""
    
    def test_create_integrated_memory_system_basic(self):
        """Test creating integrated memory system with basic parameters."""
        mock_vectorstore = Mock()
        mock_llm = Mock()
        mock_character_repo = Mock()
        mock_db_utils = Mock()
        
        with patch('memory.memory_factory.IntegratedNovelMemorySystem') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            result = create_integrated_memory_system(
                vectorstore_client=mock_vectorstore,
                llm_client=mock_llm,
                character_repo=mock_character_repo,
                db_utils=mock_db_utils
            )
            
            # Should create instance with provided parameters
            mock_class.assert_called_once_with(
                vectorstore_client=mock_vectorstore,
                llm_client=mock_llm,
                character_repo=mock_character_repo,
                db_utils=mock_db_utils,
                max_memory_tokens=32000,
                consistency_level="high"
            )
            
            assert result is mock_instance
    
    def test_create_integrated_memory_system_custom_params(self):
        """Test creating integrated memory system with custom parameters."""
        mock_vectorstore = Mock()
        mock_llm = Mock()
        
        with patch('memory.memory_factory.IntegratedNovelMemorySystem') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            result = create_integrated_memory_system(
                vectorstore_client=mock_vectorstore,
                llm_client=mock_llm,
                max_memory_tokens=16000,
                consistency_level="medium"
            )
            
            mock_class.assert_called_once_with(
                vectorstore_client=mock_vectorstore,
                llm_client=mock_llm,
                character_repo=None,
                db_utils=None,
                max_memory_tokens=16000,
                consistency_level="medium"
            )
            
            assert result is mock_instance
    
    def test_create_integrated_memory_system_no_params(self):
        """Test creating integrated memory system with no parameters."""
        with patch('memory.memory_factory.IntegratedNovelMemorySystem') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            result = create_integrated_memory_system()
            
            mock_class.assert_called_once_with(
                vectorstore_client=None,
                llm_client=None,
                character_repo=None,
                db_utils=None,
                max_memory_tokens=32000,
                consistency_level="high"
            )
            
            assert result is mock_instance
    
    def test_create_integrated_memory_system_success_logging(self):
        """Test successful creation logs appropriate message."""
        with patch('memory.memory_factory.IntegratedNovelMemorySystem') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            with patch('memory.memory_factory.logger') as mock_logger:
                result = create_integrated_memory_system()
                
                mock_logger.info.assert_called_once_with(
                    "Successfully created IntegratedNovelMemorySystem with emotional memory"
                )
    
    def test_create_integrated_memory_system_exception_fallback(self):
        """Test exception handling creates fallback system."""
        mock_vectorstore = Mock()
        mock_llm = Mock()
        mock_character_repo = Mock()
        mock_db_utils = Mock()
        
        with patch('memory.memory_factory.IntegratedNovelMemorySystem') as mock_class:
            # First call raises exception, second call succeeds
            mock_fallback = Mock()
            mock_class.side_effect = [Exception("Creation failed"), mock_fallback]
            
            with patch('memory.memory_factory.logger') as mock_logger:
                result = create_integrated_memory_system(
                    vectorstore_client=mock_vectorstore,
                    llm_client=mock_llm,
                    character_repo=mock_character_repo,
                    db_utils=mock_db_utils
                )
                
                # Should log error and warning
                mock_logger.error.assert_called_once()
                mock_logger.warning.assert_called_once_with(
                    "Creating fallback memory system without emotional memory"
                )
                
                # Should create fallback without emotional memory
                assert mock_class.call_count == 2
                fallback_call = mock_class.call_args_list[1]
                assert fallback_call[1]['db_utils'] is None
                
                assert result is mock_fallback
    
    def test_create_integrated_memory_system_exception_details(self):
        """Test exception handling logs correct error details."""
        test_error = ValueError("Test error message")
        
        with patch('memory.memory_factory.IntegratedNovelMemorySystem') as mock_class:
            mock_class.side_effect = [test_error, Mock()]
            
            with patch('memory.memory_factory.logger') as mock_logger:
                create_integrated_memory_system()
                
                # Should log the specific error
                mock_logger.error.assert_called_once_with(
                    f"Failed to create IntegratedNovelMemorySystem: {test_error}"
                )


class TestCreateEmotionalMemorySystem:
    """Test create_emotional_memory_system factory function."""
    
    def test_create_emotional_memory_system_success(self):
        """Test successful creation of emotional memory system."""
        mock_db_utils = Mock()
        
        with patch('memory.memory_factory.EmotionalMemorySystem') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            with patch('memory.memory_factory.logger') as mock_logger:
                result = create_emotional_memory_system(db_utils=mock_db_utils)
                
                mock_class.assert_called_once_with(db_utils=mock_db_utils)
                mock_logger.info.assert_called_once_with(
                    "Successfully created EmotionalMemorySystem"
                )
                
                assert result is mock_instance
    
    def test_create_emotional_memory_system_no_db_utils(self):
        """Test emotional memory system creation with no db_utils."""
        with patch('memory.memory_factory.logger') as mock_logger:
            result = create_emotional_memory_system(db_utils=None)
            
            mock_logger.warning.assert_called_once_with(
                "No database utilities provided for emotional memory system"
            )
            
            assert result is None
    
    def test_create_emotional_memory_system_exception(self):
        """Test emotional memory system creation with exception."""
        mock_db_utils = Mock()
        test_error = RuntimeError("Database connection failed")
        
        with patch('memory.memory_factory.EmotionalMemorySystem') as mock_class:
            mock_class.side_effect = test_error
            
            with patch('memory.memory_factory.logger') as mock_logger:
                result = create_emotional_memory_system(db_utils=mock_db_utils)
                
                mock_logger.error.assert_called_once_with(
                    f"Failed to create EmotionalMemorySystem: {test_error}"
                )
                
                assert result is None
    
    def test_create_emotional_memory_system_none_input(self):
        """Test emotional memory system creation with None input."""
        with patch('memory.memory_factory.logger') as mock_logger:
            result = create_emotional_memory_system()
            
            mock_logger.warning.assert_called_once_with(
                "No database utilities provided for emotional memory system"
            )
            
            assert result is None


class TestValidateMemorySystemComponents:
    """Test validate_memory_system_components function."""
    
    def test_validate_memory_system_complete(self):
        """Test validation of complete memory system."""
        mock_memory_system = Mock()
        mock_memory_system.memory_manager = Mock()
        mock_memory_system.chunker = Mock()
        mock_memory_system.consistency_manager = Mock()
        mock_memory_system.emotional_memory = Mock()
        mock_memory_system.structure_manager = Mock()
        mock_memory_system.style_manager = Mock()
        
        result = validate_memory_system_components(mock_memory_system)
        
        expected = {
            'memory_manager': True,
            'chunker': True,
            'consistency_manager': True,
            'emotional_memory': True,
            'structure_manager': True,
            'style_manager': True,
            'overall_status': 'complete'
        }
        
        assert result == expected
    
    def test_validate_memory_system_basic_only(self):
        """Test validation of memory system with only core components."""
        mock_memory_system = Mock()
        mock_memory_system.memory_manager = Mock()
        mock_memory_system.chunker = Mock()
        mock_memory_system.consistency_manager = Mock()
        mock_memory_system.emotional_memory = None
        mock_memory_system.structure_manager = None
        mock_memory_system.style_manager = None
        
        result = validate_memory_system_components(mock_memory_system)
        
        expected = {
            'memory_manager': True,
            'chunker': True,
            'consistency_manager': True,
            'emotional_memory': False,
            'structure_manager': False,
            'style_manager': False,
            'overall_status': 'basic'
        }
        
        assert result == expected
    
    def test_validate_memory_system_partial(self):
        """Test validation of memory system with some advanced components."""
        mock_memory_system = Mock()
        mock_memory_system.memory_manager = Mock()
        mock_memory_system.chunker = Mock()
        mock_memory_system.consistency_manager = Mock()
        mock_memory_system.emotional_memory = Mock()
        mock_memory_system.structure_manager = None
        mock_memory_system.style_manager = None
        
        result = validate_memory_system_components(mock_memory_system)
        
        expected = {
            'memory_manager': True,
            'chunker': True,
            'consistency_manager': True,
            'emotional_memory': True,
            'structure_manager': False,
            'style_manager': False,
            'overall_status': 'partial'
        }
        
        assert result == expected
    
    def test_validate_memory_system_failed_missing_core(self):
        """Test validation of memory system missing core components."""
        mock_memory_system = Mock()
        mock_memory_system.memory_manager = None
        mock_memory_system.chunker = Mock()
        mock_memory_system.consistency_manager = Mock()
        mock_memory_system.emotional_memory = Mock()
        mock_memory_system.structure_manager = Mock()
        mock_memory_system.style_manager = Mock()
        
        result = validate_memory_system_components(mock_memory_system)
        
        expected = {
            'memory_manager': False,
            'chunker': True,
            'consistency_manager': True,
            'emotional_memory': True,
            'structure_manager': True,
            'style_manager': True,
            'overall_status': 'failed'
        }
        
        assert result == expected
    
    def test_validate_memory_system_missing_attributes(self):
        """Test validation when memory system lacks expected attributes."""
        mock_memory_system = Mock(spec=[])  # Empty spec means no attributes
        
        result = validate_memory_system_components(mock_memory_system)
        
        expected = {
            'memory_manager': False,
            'chunker': False,
            'consistency_manager': False,
            'emotional_memory': False,
            'structure_manager': False,
            'style_manager': False,
            'overall_status': 'failed'
        }
        
        assert result == expected
    
    def test_validate_memory_system_exception_handling(self):
        """Test validation handles exceptions gracefully."""
        mock_memory_system = Mock()
        mock_memory_system.memory_manager = Mock()
        
        # Simulate exception when accessing attributes
        def side_effect(name):
            if name == 'chunker':
                raise AttributeError("Chunker access failed")
            return Mock()
        
        mock_memory_system.__getattr__ = side_effect
        
        with patch('memory.memory_factory.logger') as mock_logger:
            result = validate_memory_system_components(mock_memory_system)
            
            # Should log error
            mock_logger.error.assert_called_once()
            
            # Should have error in result
            assert 'error' in result
            assert result['overall_status'] == 'failed'
    
    def test_validate_memory_system_none_components(self):
        """Test validation with components explicitly set to None."""
        mock_memory_system = Mock()
        mock_memory_system.memory_manager = Mock()
        mock_memory_system.chunker = Mock()
        mock_memory_system.consistency_manager = Mock()
        mock_memory_system.emotional_memory = None
        mock_memory_system.structure_manager = None
        mock_memory_system.style_manager = None
        
        result = validate_memory_system_components(mock_memory_system)
        
        # Core components present, advanced components None
        assert result['memory_manager'] is True
        assert result['chunker'] is True
        assert result['consistency_manager'] is True
        assert result['emotional_memory'] is False
        assert result['structure_manager'] is False
        assert result['style_manager'] is False
        assert result['overall_status'] == 'basic'


class TestIntegrationScenarios:
    """Test integration scenarios and complex use cases."""
    
    def test_full_memory_system_creation_workflow(self):
        """Test complete workflow of creating a memory system."""
        mock_vectorstore = Mock()
        mock_llm = Mock()
        mock_character_repo = Mock()
        mock_db_utils = Mock()
        
        with patch('memory.memory_factory.IntegratedNovelMemorySystem') as mock_integrated:
            mock_memory_system = Mock()
            mock_memory_system.memory_manager = Mock()
            mock_memory_system.chunker = Mock()
            mock_memory_system.consistency_manager = Mock()
            mock_memory_system.emotional_memory = Mock()
            mock_memory_system.structure_manager = Mock()
            mock_memory_system.style_manager = Mock()
            mock_integrated.return_value = mock_memory_system
            
            # Create the system
            memory_system = create_integrated_memory_system(
                vectorstore_client=mock_vectorstore,
                llm_client=mock_llm,
                character_repo=mock_character_repo,
                db_utils=mock_db_utils,
                max_memory_tokens=16000,
                consistency_level="medium"
            )
            
            # Validate the system
            validation_result = validate_memory_system_components(memory_system)
            
            # Should be complete system
            assert validation_result['overall_status'] == 'complete'
            assert all(validation_result[comp] for comp in [
                'memory_manager', 'chunker', 'consistency_manager',
                'emotional_memory', 'structure_manager', 'style_manager'
            ])
    
    def test_fallback_system_creation_and_validation(self):
        """Test fallback system creation and validation."""
        mock_vectorstore = Mock()
        mock_llm = Mock()
        
        with patch('memory.memory_factory.IntegratedNovelMemorySystem') as mock_integrated:
            # First call fails, second succeeds with fallback
            mock_fallback_system = Mock()
            mock_fallback_system.memory_manager = Mock()
            mock_fallback_system.chunker = Mock()
            mock_fallback_system.consistency_manager = Mock()
            mock_fallback_system.emotional_memory = None  # Fallback has no emotional memory
            mock_fallback_system.structure_manager = None
            mock_fallback_system.style_manager = None
            
            mock_integrated.side_effect = [
                Exception("Initial creation failed"),
                mock_fallback_system
            ]
            
            with patch('memory.memory_factory.logger'):
                # Create system (should fallback)
                memory_system = create_integrated_memory_system(
                    vectorstore_client=mock_vectorstore,
                    llm_client=mock_llm,
                    db_utils=Mock()  # This should be disabled in fallback
                )
                
                # Validate fallback system
                validation_result = validate_memory_system_components(memory_system)
                
                # Should be basic system
                assert validation_result['overall_status'] == 'basic'
                assert validation_result['memory_manager'] is True
                assert validation_result['chunker'] is True
                assert validation_result['consistency_manager'] is True
                assert validation_result['emotional_memory'] is False
    
    def test_emotional_memory_creation_with_validation(self):
        """Test creating emotional memory system and validating integration."""
        mock_db_utils = Mock()
        
        with patch('memory.memory_factory.EmotionalMemorySystem') as mock_emotional:
            mock_emotional_instance = Mock()
            mock_emotional.return_value = mock_emotional_instance
            
            # Create emotional memory system
            emotional_system = create_emotional_memory_system(db_utils=mock_db_utils)
            
            assert emotional_system is mock_emotional_instance
            
            # Test it could be integrated into main system
            with patch('memory.memory_factory.IntegratedNovelMemorySystem') as mock_integrated:
                mock_main_system = Mock()
                mock_main_system.emotional_memory = emotional_system
                mock_main_system.memory_manager = Mock()
                mock_main_system.chunker = Mock()
                mock_main_system.consistency_manager = Mock()
                mock_main_system.structure_manager = None
                mock_main_system.style_manager = None
                mock_integrated.return_value = mock_main_system
                
                main_system = create_integrated_memory_system(db_utils=mock_db_utils)
                validation_result = validate_memory_system_components(main_system)
                
                # Should have emotional memory
                assert validation_result['emotional_memory'] is True
                assert validation_result['overall_status'] == 'partial'
    
    def test_memory_system_creation_logging_flow(self):
        """Test comprehensive logging throughout memory system creation."""
        with patch('memory.memory_factory.logger') as mock_logger:
            with patch('memory.memory_factory.IntegratedNovelMemorySystem') as mock_class:
                mock_instance = Mock()
                mock_class.return_value = mock_instance
                
                # Test successful creation
                create_integrated_memory_system()
                
                # Should log success
                mock_logger.info.assert_called_with(
                    "Successfully created IntegratedNovelMemorySystem with emotional memory"
                )
                
                # Reset logger
                mock_logger.reset_mock()
                
                # Test emotional memory creation
                with patch('memory.memory_factory.EmotionalMemorySystem') as mock_emotional:
                    mock_emotional_instance = Mock()
                    mock_emotional.return_value = mock_emotional_instance
                    
                    create_emotional_memory_system(Mock())
                    
                    mock_logger.info.assert_called_with(
                        "Successfully created EmotionalMemorySystem"
                    )
    
    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling across all factory functions."""
        # Test various types of exceptions
        exceptions_to_test = [
            ValueError("Configuration error"),
            RuntimeError("Runtime error"),
            ImportError("Module not found"),
            AttributeError("Attribute missing"),
            TypeError("Type error"),
        ]
        
        for exception in exceptions_to_test:
            with patch('memory.memory_factory.IntegratedNovelMemorySystem') as mock_class:
                mock_class.side_effect = [exception, Mock()]  # Fail then succeed
                
                with patch('memory.memory_factory.logger') as mock_logger:
                    result = create_integrated_memory_system()
                    
                    # Should log the specific error
                    mock_logger.error.assert_called_once()
                    error_call = mock_logger.error.call_args[0][0]
                    assert str(exception) in error_call
                    
                    # Should create fallback
                    mock_logger.warning.assert_called_once()
                    
                    # Should return fallback system
                    assert result is not None


class TestMemoryFactoryEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_create_memory_system_with_all_none_parameters(self):
        """Test creating memory system with all None parameters."""
        with patch('memory.memory_factory.IntegratedNovelMemorySystem') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            result = create_integrated_memory_system(
                vectorstore_client=None,
                llm_client=None,
                character_repo=None,
                db_utils=None,
                max_memory_tokens=0,  # Edge case
                consistency_level=""  # Edge case
            )
            
            mock_class.assert_called_once_with(
                vectorstore_client=None,
                llm_client=None,
                character_repo=None,
                db_utils=None,
                max_memory_tokens=0,
                consistency_level=""
            )
            
            assert result is mock_instance
    
    def test_validate_memory_system_with_invalid_object(self):
        """Test validation with invalid memory system object."""
        # Test with non-mock object
        invalid_system = "not a memory system"
        
        result = validate_memory_system_components(invalid_system)
        
        # Should handle gracefully
        assert result['overall_status'] == 'failed'
        assert all(not result[comp] for comp in [
            'memory_manager', 'chunker', 'consistency_manager',
            'emotional_memory', 'structure_manager', 'style_manager'
        ])
    
    def test_memory_system_partial_attribute_access_failure(self):
        """Test validation when some attribute access fails."""
        mock_system = Mock()
        mock_system.memory_manager = Mock()
        mock_system.chunker = Mock()
        
        # Make consistency_manager access raise exception
        type(mock_system).consistency_manager = PropertyMock(
            side_effect=RuntimeError("Access failed")
        )
        
        with patch('memory.memory_factory.logger') as mock_logger:
            result = validate_memory_system_components(mock_system)
            
            # Should log error and fail validation
            mock_logger.error.assert_called_once()
            assert result['overall_status'] == 'failed'
            assert 'error' in result
    
    def test_create_emotional_memory_system_with_invalid_db_utils(self):
        """Test emotional memory creation with invalid db_utils."""
        invalid_db_utils = "not a db utils object"
        
        with patch('memory.memory_factory.EmotionalMemorySystem') as mock_class:
            # Simulate that invalid db_utils causes creation to fail
            mock_class.side_effect = TypeError("Invalid db_utils type")
            
            with patch('memory.memory_factory.logger') as mock_logger:
                result = create_emotional_memory_system(db_utils=invalid_db_utils)
                
                mock_logger.error.assert_called_once()
                assert result is None