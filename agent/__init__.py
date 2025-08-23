"""Agent package for agentic RAG with knowledge graph."""

__version__ = "0.1.0"

# Import creative tools to register them with the agent
try:
    from . import creative_agent_tools
    from . import plot_analysis_tools
    from . import validation_tools
    from . import performance_monitoring_tools
    from . import genre_specific_tools
except ImportError:
    pass  # Creative tools are optional

# Initialize advanced features
try:
    from .creative_performance_monitor import creative_monitor
    from .memory_optimization import novel_memory_manager
    from .testing_framework import novel_test_framework
except ImportError:
    pass  # Advanced features are optional