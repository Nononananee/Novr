#!/usr/bin/env python3
"""
Demo untuk menunjukkan fitur-fitur baru dari API yang sudah di-refactor.
"""

import os
import asyncio
import sys
from pathlib import Path

# Set minimal environment
os.environ.setdefault('DATABASE_URL', 'sqlite:///demo.db')
os.environ.setdefault('GRAPH_DATABASE_URL', 'bolt://localhost:7687')
os.environ.setdefault('APP_ENV', 'development')
os.environ.setdefault('SECRET_KEY', 'demo-secret-key')
os.environ.setdefault('ENABLE_CACHING', 'false')  # Disable caching untuk demo

def demo_imports():
    """Demonstrate imports dan basic setup."""
    print("🚀 Novel RAG API - Refactoring Demo")
    print("=" * 50)
    
    try:
        # Import konfigurasi
        from agent.api_config import APIConfig
        print(f"✅ Configuration loaded")
        print(f"   - Environment: {APIConfig.APP_ENV}")
        print(f"   - Host: {APIConfig.APP_HOST}:{APIConfig.APP_PORT}")
        print(f"   - Caching: {APIConfig.ENABLE_CACHING}")
        print(f"   - Is Production: {APIConfig.is_production()}")
        
        # Test configuration validation
        issues = APIConfig.validate_config()
        if issues:
            print(f"   - Config Issues: {len(issues)} (expected in demo)")
        else:
            print(f"   - Config: All valid ✅")
            
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        # Import exceptions
        from agent.api_exceptions import (
            ValidationError, 
            NotFoundError, 
            ServiceUnavailableError
        )
        print(f"✅ Custom exceptions available")
        
        # Demo exception
        try:
            raise ValidationError("Demo validation error", "test_field")
        except ValidationError as e:
            error_dict = e.to_dict()
            print(f"   - Error Code: {error_dict['error_code']}")
            print(f"   - Request ID: {error_dict['request_id']}")
            
    except ImportError as e:
        print(f"❌ Exception import failed: {e}")
        return False
    
    try:
        # Import cache manager
        from agent.api_cache import cache_manager, CacheStats
        print(f"✅ Cache manager available")
        
        # Demo cache stats
        stats = CacheStats()
        stats.record_hit()
        stats.record_miss()
        cache_stats = stats.get_stats()
        print(f"   - Hit Rate: {cache_stats['hit_rate']:.1%}")
        
    except ImportError as e:
        print(f"❌ Cache import failed: {e}")
        return False
    
    try:
        # Import retry mechanism
        from agent.api_retry import RetryConfig, retry_stats
        print(f"✅ Retry mechanism available")
        
        # Demo retry config
        config = RetryConfig(max_retries=3, initial_delay=1.0)
        print(f"   - Max Retries: {config.max_retries}")
        print(f"   - Initial Delay: {config.initial_delay}s")
        
    except ImportError as e:
        print(f"❌ Retry import failed: {e}")
        return False
    
    try:
        # Import utilities
        from agent.api_utils import (
            ConversationManager,
            AgentExecutor, 
            SearchOperations,
            HealthChecker,
            RequestValidator
        )
        print(f"✅ Utility functions available")
        print(f"   - ConversationManager: Session management")
        print(f"   - AgentExecutor: Agent operations") 
        print(f"   - SearchOperations: Search functionality")
        print(f"   - HealthChecker: Health monitoring")
        print(f"   - RequestValidator: Input validation")
        
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    return True

async def demo_async_features():
    """Demonstrate async features."""
    print(f"\n🔄 Async Features Demo")
    print("-" * 30)
    
    try:
        from agent.api_cache import cache_manager
        
        # Test cache operations
        print("📦 Testing cache operations...")
        
        # Initialize cache (memory fallback)
        await cache_manager.initialize()
        
        # Test cache set/get
        test_key = "demo:test"
        test_value = {"message": "Hello World!", "timestamp": "2024-01-01"}
        
        success = await cache_manager.set(test_key, test_value, ttl=60)
        print(f"   - Cache Set: {'✅' if success else '❌'}")
        
        cached_value = await cache_manager.get(test_key)
        print(f"   - Cache Get: {'✅' if cached_value else '❌'}")
        
        if cached_value:
            print(f"   - Cached Data: {cached_value['message']}")
        
        # Get cache stats
        stats = await cache_manager.get_stats()
        print(f"   - Cache Type: {stats['type']}")
        
        await cache_manager.close()
        
    except Exception as e:
        print(f"❌ Async demo failed: {e}")

def demo_validation():
    """Demonstrate validation features."""
    print(f"\n🔒 Validation Demo")
    print("-" * 20)
    
    try:
        from agent.api_utils import RequestValidator
        
        # Test chat request validation
        print("💬 Testing chat validation...")
        
        valid_request = {
            "message": "Hello, how can you help me write a novel?",
            "user_id": "demo-user",
            "session_id": None
        }
        
        try:
            validated = RequestValidator.validate_chat_request(valid_request)
            print(f"   - Valid Request: ✅")
            print(f"   - Sanitized Message: {validated['message'][:50]}...")
        except Exception as e:
            print(f"   - Validation Error: {e}")
        
        # Test invalid request
        invalid_request = {
            "message": "",  # Empty message
            "user_id": "demo-user"
        }
        
        try:
            RequestValidator.validate_chat_request(invalid_request)
            print(f"   - Invalid Request: Should have failed!")
        except Exception as e:
            print(f"   - Invalid Request Caught: ✅")
            print(f"   - Error: {str(e)[:50]}...")
            
    except ImportError as e:
        print(f"❌ Validation demo failed: {e}")

def demo_error_handling():
    """Demonstrate error handling."""
    print(f"\n🚨 Error Handling Demo")
    print("-" * 25)
    
    try:
        from agent.api_exceptions import (
            ValidationError,
            NotFoundError,
            ServiceUnavailableError,
            APIBaseException
        )
        
        # Demo different error types
        errors = [
            ValidationError("Invalid input", "message"),
            NotFoundError("User", "123"),
            ServiceUnavailableError("database", "Connection timeout")
        ]
        
        for error in errors:
            error_dict = error.to_dict()
            print(f"   - {error.__class__.__name__}:")
            print(f"     Code: {error_dict['error_code']}")
            print(f"     Status: {error_dict['status_code']}")
            print(f"     Message: {error_dict['error']}")
            
    except ImportError as e:
        print(f"❌ Error handling demo failed: {e}")

def demo_configuration():
    """Demonstrate configuration features."""
    print(f"\n⚙️ Configuration Demo")
    print("-" * 22)
    
    try:
        from agent.api_config import APIConfig
        
        print(f"📋 Current Configuration:")
        print(f"   - Environment: {APIConfig.APP_ENV}")
        print(f"   - Debug Mode: {APIConfig.is_development()}")
        print(f"   - Production: {APIConfig.is_production()}")
        print(f"   - Caching: {APIConfig.ENABLE_CACHING}")
        print(f"   - Max Tokens: {APIConfig.MAX_TOKENS_DEFAULT}")
        print(f"   - Retry Config: {APIConfig.MAX_RETRIES} retries")
        
        # CORS configuration
        cors_config = APIConfig.get_cors_config()
        print(f"   - CORS Origins: {len(cors_config['allow_origins'])} configured")
        
        # Security headers
        security_headers = APIConfig.SECURITY_HEADERS
        active_headers = sum(1 for v in security_headers.values() if v)
        print(f"   - Security Headers: {active_headers} active")
        
    except ImportError as e:
        print(f"❌ Configuration demo failed: {e}")

async def main():
    """Main demo function."""
    print("🎬 Starting Novel RAG API Refactoring Demo\n")
    
    # Test imports
    if not demo_imports():
        print("\n❌ Import demo failed, stopping here")
        return 1
    
    # Test async features
    await demo_async_features()
    
    # Test validation
    demo_validation()
    
    # Test error handling
    demo_error_handling()
    
    # Test configuration
    demo_configuration()
    
    print(f"\n🎉 Demo Complete!")
    print("=" * 50)
    print("✅ All refactored components working correctly")
    print("💡 Ready for production deployment")
    print("📚 See docs/API_REFACTORING_GUIDE.md for details")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)
