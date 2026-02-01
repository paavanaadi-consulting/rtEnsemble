#!/usr/bin/env python3
"""
Monitor training progress
"""

import os
import time
import subprocess
from pathlib import Path
from datetime import datetime

def check_process_status():
    """Check if training process is running"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        
        for line in result.stdout.split('\n'):
            if 'fetch_and_train.py' in line and 'grep' not in line:
                parts = line.split()
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                time_str = parts[9]
                return {
                    'running': True,
                    'pid': pid,
                    'cpu': cpu,
                    'mem': mem,
                    'time': time_str
                }
        
        return {'running': False}
    except Exception as e:
        print(f"Error checking process: {e}")
        return {'running': False}

def check_model_files():
    """Check for model files"""
    models_dir = Path('/Users/ashokasangapallar/Desktop/studyrepos/tradingproject/rtEnsemble/models')
    
    model_files = []
    for symbol_dir in models_dir.glob('*/'):
        if symbol_dir.is_dir():
            symbol = symbol_dir.name
            files = list(symbol_dir.glob('*.pth')) + list(symbol_dir.glob('*.pkl')) + list(symbol_dir.glob('*.json'))
            if files:
                model_files.append({
                    'symbol': symbol,
                    'files': len(files),
                    'latest': max(files, key=lambda f: f.stat().st_mtime).stat().st_mtime
                })
    
    return model_files

def main():
    """Main monitoring function"""
    print("\n" + "="*60)
    print("🔍 Training Monitor")
    print("="*60 + "\n")
    
    # Check process
    status = check_process_status()
    
    if status['running']:
        print(f"✅ Training process is RUNNING")
        print(f"   PID: {status['pid']}")
        print(f"   CPU: {status['cpu']}%")
        print(f"   Memory: {status['mem']}%")
        print(f"   Runtime: {status['time']}")
    else:
        print("❌ Training process is NOT running")
    
    print()
    
    # Check model files
    model_files = check_model_files()
    
    if model_files:
        print("📁 Model Files Found:")
        for model_info in model_files:
            latest_time = datetime.fromtimestamp(model_info['latest'])
            print(f"   {model_info['symbol']}: {model_info['files']} files (latest: {latest_time.strftime('%Y-%m-%d %H:%M:%S')})")
    else:
        print("📁 No symbol-specific model directories found yet")
    
    # Check for any .pth or .pkl files in models root
    models_dir = Path('/Users/ashokasangapallar/Desktop/studyrepos/tradingproject/rtEnsemble/models')
    root_files = list(models_dir.glob('*.pth')) + list(models_dir.glob('*.pkl'))
    
    if root_files:
        print(f"\n   {len(root_files)} files in models root directory")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
