import pynvml
import psutil
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class GPUMetrics:
    memory_used: int
    memory_total: int
    gpu_utilization: int
    temperature: int

class HardwareMonitor:
    def __init__(self):
        pynvml.nvmlInit()
        self.gpu_count = pynvml.nvmlDeviceGetCount()
        self.handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) 
            for i in range(self.gpu_count)
        ]
    
    def get_gpu_metrics(self) -> Dict[int, GPUMetrics]:
        metrics = {}
        for idx, handle in enumerate(self.handles):
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            
            metrics[idx] = GPUMetrics(
                memory_used=memory.used,
                memory_total=memory.total,
                gpu_utilization=utilization.gpu,
                temperature=temperature
            )
        return metrics
