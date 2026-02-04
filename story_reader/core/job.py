"""
Job tracking with file locking for concurrent access safety.
"""

import json
import fcntl
from pathlib import Path
from datetime import datetime
from typing import Optional


class Job:
    """
    Tracks the status of a pipeline step with file-based persistence.
    
    Uses file locking to prevent race conditions when multiple
    processes access the same jobs file.
    """
    
    def __init__(self, name: str, tracker_file: Path):
        """
        Initialize a job tracker.
        
        Args:
            name: Unique name for this job (e.g., "transcription")
            tracker_file: Path to the JSON file for tracking all jobs
        """
        self.name = name
        self.tracker_file = tracker_file
        self.status = "pending"
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.error_message: Optional[str] = None
        self._save_status()
    
    def start(self) -> "Job":
        """Mark job as running. Returns self for chaining."""
        self.status = "running"
        self.start_time = datetime.now().isoformat()
        self._save_status()
        print(f"[{self.name}] started...")
        return self
    
    def complete(self) -> "Job":
        """Mark job as completed. Returns self for chaining."""
        self.status = "completed"
        self.end_time = datetime.now().isoformat()
        self._save_status()
        print(f"[{self.name}] completed.")
        return self
    
    def fail(self, error_msg: str) -> "Job":
        """Mark job as failed with error message. Returns self for chaining."""
        self.status = "failed"
        self.error_message = error_msg
        self.end_time = datetime.now().isoformat()
        self._save_status()
        print(f"[{self.name}] FAILED: {error_msg}")
        return self
    
    def skip(self, reason: str = "cached") -> "Job":
        """Mark job as skipped. Returns self for chaining."""
        self.status = f"skipped: {reason}"
        self.end_time = datetime.now().isoformat()
        self._save_status()
        print(f"[{self.name}] skipped ({reason})")
        return self
    
    def _save_status(self) -> None:
        """Save job status with file locking to prevent race conditions."""
        lock_file = self.tracker_file.with_suffix(".lock")
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(lock_file, "w") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                all_jobs = {}
                if self.tracker_file.exists():
                    with open(self.tracker_file, "r") as f:
                        try:
                            all_jobs = json.load(f)
                        except json.JSONDecodeError:
                            all_jobs = {}
                
                all_jobs[self.name] = {
                    "status": self.status,
                    "start_time": self.start_time,
                    "end_time": self.end_time,
                    "error_message": self.error_message,
                }
                
                with open(self.tracker_file, "w") as f:
                    json.dump(all_jobs, f, indent=2)
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
    
    def to_dict(self) -> dict:
        """Return job status as dictionary."""
        return {
            "name": self.name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error_message": self.error_message,
        }
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate job duration in seconds, if completed."""
        if self.start_time and self.end_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            return (end - start).total_seconds()
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if job completed successfully."""
        return self.status == "completed"
    
    @property
    def is_failed(self) -> bool:
        """Check if job failed."""
        return self.status == "failed"
    
    def __repr__(self) -> str:
        return f"Job(name='{self.name}', status='{self.status}')"


class JobManager:
    """
    Manages multiple jobs for a pipeline run.
    """
    
    def __init__(self, tracker_file: Path):
        self.tracker_file = tracker_file
        self.jobs: dict[str, Job] = {}
    
    def create_job(self, name: str) -> Job:
        """Create and track a new job."""
        job = Job(name, self.tracker_file)
        self.jobs[name] = job
        return job
    
    def get_job(self, name: str) -> Optional[Job]:
        """Get an existing job by name."""
        return self.jobs.get(name)
    
    def load_previous_jobs(self) -> dict:
        """Load job statuses from previous runs."""
        if self.tracker_file.exists():
            with open(self.tracker_file, "r") as f:
                return json.load(f)
        return {}
    
    def clear(self) -> None:
        """Clear all job records."""
        if self.tracker_file.exists():
            self.tracker_file.unlink()
        self.jobs.clear()
