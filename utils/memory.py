import torch
import gc
import os
import psutil
import logging
from rich.logging import RichHandler

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("memory_utils")

class MemoryTracker:
    """Track and optimize memory usage during training"""
    
    def __init__(self, threshold_pct=85, log_interval=10):
        self.threshold_pct = threshold_pct
        self.log_interval = log_interval
        self.step_counter = 0
        self.peak_memory = 0
        
    def update(self, force_gc=False):
        """Update memory stats and perform garbage collection if needed"""
        self.step_counter += 1
        
        # Only check periodically to minimize overhead
        if self.step_counter % self.log_interval != 0 and not force_gc:
            return {}
            
        memory_stats = {}
        
        # System memory
        process = psutil.Process(os.getpid())
        system_used = process.memory_info().rss / (1024 * 1024)  # MB
        system_percent = process.memory_percent()
        memory_stats['system_used_mb'] = system_used
        memory_stats['system_percent'] = system_percent
        
        # PyTorch memory
        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            gpu_percent = 100 * gpu_used / gpu_total
            memory_stats['gpu_used_mb'] = gpu_used
            memory_stats['gpu_percent'] = gpu_percent
            
            # Track peak memory
            if gpu_used > self.peak_memory:
                self.peak_memory = gpu_used
                memory_stats['peak_memory_mb'] = self.peak_memory
                
        elif torch.backends.mps.is_available():
            try:
                mps_used = torch.mps.current_allocated_memory() / (1024 * 1024)  # MB
                memory_stats['mps_used_mb'] = mps_used
                
                # Track peak memory
                if mps_used > self.peak_memory:
                    self.peak_memory = mps_used
                    memory_stats['peak_memory_mb'] = self.peak_memory
            except:
                # Fall back if MPS memory tracking fails
                memory_stats['mps_used_mb'] = 0
                
        # Check if memory usage is above threshold
        if (torch.cuda.is_available() and memory_stats.get('gpu_percent', 0) > self.threshold_pct) or \
           (system_percent > self.threshold_pct):
            self._clean_memory()
            memory_stats['gc_triggered'] = True
        
        # Force garbage collection if requested
        if force_gc:
            self._clean_memory()
            memory_stats['gc_triggered'] = True
            
        return memory_stats
    
    def _clean_memory(self):
        """Perform garbage collection and free unused memory"""
        # Run Python's garbage collector
        gc.collect()
        
        # Free PyTorch memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        logger.debug("Memory cleaned")
            
    def get_stats(self):
        """Get current memory statistics"""
        return self.update(force_gc=False)
        
    def log_memory(self, prefix=""):
        """Log memory statistics"""
        stats = self.get_stats()
        
        if stats:
            if torch.cuda.is_available():
                logger.info(f"{prefix} GPU Memory: {stats.get('gpu_used_mb', 0):.1f}MB "
                          f"({stats.get('gpu_percent', 0):.1f}%), "
                          f"System: {stats.get('system_used_mb', 0):.1f}MB "
                          f"({stats.get('system_percent', 0):.1f}%)")
            elif torch.backends.mps.is_available():
                logger.info(f"{prefix} MPS Memory: {stats.get('mps_used_mb', 0):.1f}MB, "
                          f"System: {stats.get('system_used_mb', 0):.1f}MB "
                          f"({stats.get('system_percent', 0):.1f}%)")
            else:
                logger.info(f"{prefix} System Memory: {stats.get('system_used_mb', 0):.1f}MB "
                          f"({stats.get('system_percent', 0):.1f}%)")
        
        return stats

# Create a global tracker instance
memory_tracker = MemoryTracker()

def log_memory(prefix=""):
    """Convenience function to log memory usage"""
    return memory_tracker.log_memory(prefix)

def clean_memory():
    """Force garbage collection and memory cleanup"""
    memory_tracker.update(force_gc=True)