#!/usr/bin/env python3
"""
AutoDL Training Monitor for YOLOv11 + MaxViT
Monitors GPU usage, training progress, and system resources
"""

import psutil
import GPUtil
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

class AutoDLMonitor:
    def __init__(self):
        self.log_file = "training_monitor.log"
        self.metrics = {
            'timestamp': [],
            'gpu_usage': [],
            'gpu_memory': [],
            'cpu_usage': [],
            'ram_usage': [],
            'disk_usage': []
        }
    
    def log_system_stats(self):
        """Log current system statistics"""
        try:
            # GPU stats
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_usage = gpu.load * 100
                gpu_memory = gpu.memoryUtil * 100
            else:
                gpu_usage = 0
                gpu_memory = 0
            
            # CPU and RAM stats
            cpu_usage = psutil.cpu_percent()
            ram = psutil.virtual_memory()
            ram_usage = ram.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Store metrics
            self.metrics['timestamp'].append(time.time())
            self.metrics['gpu_usage'].append(gpu_usage)
            self.metrics['gpu_memory'].append(gpu_memory)
            self.metrics['cpu_usage'].append(cpu_usage)
            self.metrics['ram_usage'].append(ram_usage)
            self.metrics['disk_usage'].append(disk_usage)
            
            # Print current stats
            print(f"🖥️  GPU: {gpu_usage:.1f}% | GPU Memory: {gpu_memory:.1f}%")
            print(f"💻 CPU: {cpu_usage:.1f}% | RAM: {ram_usage:.1f}% | Disk: {disk_usage:.1f}%")
            
            # Log to file
            with open(self.log_file, 'a') as f:
                f.write(f"{time.time()},{gpu_usage:.1f},{gpu_memory:.1f},{cpu_usage:.1f},{ram_usage:.1f},{disk_usage:.1f}\n")
                
        except Exception as e:
            print(f"Error logging stats: {e}")
    
    def monitor_training(self, interval=30):
        """Monitor training progress"""
        print("🔍 Starting AutoDL Training Monitor...")
        print("📊 Monitoring system resources every 30 seconds")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                self.log_system_stats()
                self.check_training_progress()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
            self.generate_report()
    
    def check_training_progress(self):
        """Check training progress from log files"""
        # Check for training logs
        runs_dir = Path("runs/train")
        if runs_dir.exists():
            latest_run = max(runs_dir.glob("*"), key=os.path.getctime, default=None)
            if latest_run:
                results_file = latest_run / "results.csv"
                if results_file.exists():
                    try:
                        df = pd.read_csv(results_file)
                        if len(df) > 0:
                            latest = df.iloc[-1]
                            print(f"📈 Epoch {latest['epoch']}: mAP50={latest.get('metrics/mAP50(B)', 'N/A'):.4f}")
                    except Exception as e:
                        print(f"Error reading training progress: {e}")
    
    def generate_report(self):
        """Generate monitoring report"""
        if not self.metrics['timestamp']:
            print("No data to generate report")
            return
        
        try:
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('AutoDL Training Monitoring Report')
            
            # GPU Usage
            axes[0,0].plot(self.metrics['gpu_usage'], label='GPU Usage %')
            axes[0,0].plot(self.metrics['gpu_memory'], label='GPU Memory %')
            axes[0,0].set_title('GPU Utilization')
            axes[0,0].legend()
            axes[0,0].set_ylabel('Percentage')
            
            # CPU Usage
            axes[0,1].plot(self.metrics['cpu_usage'], label='CPU Usage %', color='orange')
            axes[0,1].set_title('CPU Utilization')
            axes[0,1].set_ylabel('Percentage')
            
            # RAM Usage
            axes[1,0].plot(self.metrics['ram_usage'], label='RAM Usage %', color='green')
            axes[1,0].set_title('RAM Utilization')
            axes[1,0].set_ylabel('Percentage')
            
            # Disk Usage
            axes[1,1].plot(self.metrics['disk_usage'], label='Disk Usage %', color='red')
            axes[1,1].set_title('Disk Utilization')
            axes[1,1].set_ylabel('Percentage')
            
            plt.tight_layout()
            plt.savefig('monitoring_report.png', dpi=300, bbox_inches='tight')
            print("📊 Monitoring report saved as monitoring_report.png")
            
        except Exception as e:
            print(f"Error generating report: {e}")

if __name__ == "__main__":
    monitor = AutoDLMonitor()
    monitor.monitor_training()
