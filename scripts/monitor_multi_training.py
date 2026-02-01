#!/usr/bin/env python3
"""
Monitor multi-symbol training progress
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime

def check_training_process():
    """Check if training is running"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        
        for line in result.stdout.split('\n'):
            if 'train_multiple_symbols' in line and 'grep' not in line:
                parts = line.split()
                return {
                    'running': True,
                    'pid': parts[1],
                    'cpu': parts[2],
                    'mem': parts[3],
                    'time': parts[9]
                }
        
        return {'running': False}
    except Exception as e:
        print(f"Error: {e}")
        return {'running': False}

def count_trained_models():
    """Count how many symbols have trained models"""
    models_dir = Path('models')
    if not models_dir.exists():
        return []
    
    trained = []
    for symbol_dir in models_dir.iterdir():
        if symbol_dir.is_dir() and symbol_dir.name != 'checkpoints':
            # Check if it has model files
            model_files = list(symbol_dir.glob('*.pth')) + list(symbol_dir.glob('*.pkl')) + list(symbol_dir.glob('*.txt')) + list(symbol_dir.glob('*.json'))
            if len(model_files) >= 5:  # Should have at least 5 model files
                trained.append(symbol_dir.name)
    
    return sorted(trained)

def main():
    print("\n" + "="*70)
    print("🔍 MULTI-SYMBOL TRAINING MONITOR")
    print("="*70)
    print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check process
    status = check_training_process()
    
    if status['running']:
        print(f"✅ Training is RUNNING")
        print(f"   PID:     {status['pid']}")
        print(f"   CPU:     {status['cpu']}%")
        print(f"   Memory:  {status['mem']}%")
        print(f"   Runtime: {status['time']}")
    else:
        print("❌ Training is NOT running")
    
    print()
    
    # Count trained models
    trained_symbols = count_trained_models()
    
    print(f"📦 Trained Models: {len(trained_symbols)}/27")
    
    if trained_symbols:
        print("\nCompleted symbols:")
        for i, symbol in enumerate(trained_symbols, 1):
            print(f"   {i:2d}. {symbol}")
    
    # Read symbols.txt to show pending
    try:
        with open('config/symbols.txt', 'r') as f:
            all_symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        pending = [s for s in all_symbols if s not in trained_symbols]
        
        if pending:
            print(f"\n⏳ Pending symbols ({len(pending)}):")
            print(f"   {', '.join(pending)}")
    except:
        pass
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
