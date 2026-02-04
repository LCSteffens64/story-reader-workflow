"""
Abstract base class for pipeline steps.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar, Generic

from ..config import PipelineConfig
from ..core.job import Job
from ..core.cache import CacheManager

# Type variables for input/output types
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class PipelineStep(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for all pipeline steps.
    
    Each step in the pipeline inherits from this class and implements
    the `run` method to perform its specific task.
    
    Attributes:
        name: Unique identifier for this step
        config: Pipeline configuration
        cache: Cache manager for storing/retrieving cached results
    """
    
    name: str = "base_step"
    description: str = "Base pipeline step"
    
    def __init__(self, config: PipelineConfig, cache: CacheManager):
        """
        Initialize the pipeline step.
        
        Args:
            config: Pipeline configuration
            cache: Cache manager instance
        """
        self.config = config
        self.cache = cache
        self._job: Job = None
    
    @abstractmethod
    def run(self, input_data: InputT) -> OutputT:
        """
        Execute the step's main logic.
        
        This method must be implemented by subclasses to perform
        the actual work of the step.
        
        Args:
            input_data: Input data from previous step
            
        Returns:
            Output data to pass to next step
        """
        pass
    
    def execute(self, input_data: InputT) -> OutputT:
        """
        Execute the step with job tracking.
        
        This wrapper method handles job creation, status tracking,
        and error handling around the actual step execution.
        
        Args:
            input_data: Input data from previous step
            
        Returns:
            Output data to pass to next step
            
        Raises:
            Exception: Re-raises any exception from run() after marking job as failed
        """
        self._job = Job(self.name, self.config.jobs_file)
        try:
            self._job.start()
            result = self.run(input_data)
            self._job.complete()
            return result
        except Exception as e:
            self._job.fail(str(e))
            raise
    
    @property
    def job(self) -> Job:
        """Get the current job tracker."""
        return self._job
    
    def skip(self, reason: str = "cached") -> None:
        """Mark the current job as skipped."""
        if self._job:
            self._job.skip(reason)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class CompositeStep(PipelineStep[InputT, OutputT]):
    """
    A step that combines multiple sub-steps.
    
    Useful for grouping related operations that should be
    tracked as a single logical unit.
    """
    
    def __init__(self, config: PipelineConfig, cache: CacheManager, steps: list):
        super().__init__(config, cache)
        self.steps = steps
    
    def run(self, input_data: InputT) -> OutputT:
        """Run all sub-steps in sequence."""
        result = input_data
        for step in self.steps:
            result = step.execute(result)
        return result
