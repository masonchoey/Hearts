#!/usr/bin/env python3
"""
Verification script to check if the Hearts game setup is correct
"""
import sys
import os
from pathlib import Path

def print_status(message, status):
    """Print colored status message"""
    colors = {
        'ok': '\033[92m✓\033[0m',
        'error': '\033[91m✗\033[0m',
        'warn': '\033[93m⚠\033[0m',
        'info': '\033[94mℹ\033[0m'
    }
    print(f"{colors.get(status, '')} {message}")

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_status(f"Python version {version.major}.{version.minor}.{version.micro}", "ok")
        return True
    else:
        print_status(f"Python version {version.major}.{version.minor}.{version.micro} (requires 3.8+)", "error")
        return False

def check_dependencies():
    """Check if required Python packages are installed"""
    required = [
        'fastapi',
        'uvicorn',
        'pyspiel',
        'ray',
        'numpy',
        'pydantic'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print_status(f"Package '{package}' installed", "ok")
        except ImportError:
            print_status(f"Package '{package}' not found", "error")
            missing.append(package)
    
    return len(missing) == 0

def check_directory_structure():
    """Check if directory structure is correct"""
    required_dirs = [
        'backend',
        'backend/game',
        'backend/models',
        'backend/schemas',
        'frontend',
        'frontend/src',
        'frontend/src/components',
        'frontend/src/hooks',
        'frontend/src/api'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print_status(f"Directory '{dir_path}' exists", "ok")
        else:
            print_status(f"Directory '{dir_path}' not found", "error")
            all_exist = False
    
    return all_exist

def check_env_file():
    """Check if .env file exists and has CHECKPOINT_PATH"""
    if not Path('.env').exists():
        print_status(".env file not found", "warn")
        print_status("Run: cp .env.example .env", "info")
        return False
    
    print_status(".env file exists", "ok")
    
    with open('.env', 'r') as f:
        content = f.read()
        if 'CHECKPOINT_PATH=' in content:
            # Check if it's set to a value
            for line in content.split('\n'):
                if line.startswith('CHECKPOINT_PATH=') and len(line.strip()) > len('CHECKPOINT_PATH='):
                    checkpoint_path = line.split('=', 1)[1].strip()
                    if checkpoint_path and Path(checkpoint_path).exists():
                        print_status(f"CHECKPOINT_PATH is set and exists: {checkpoint_path}", "ok")
                        return True
                    else:
                        print_status(f"CHECKPOINT_PATH is set but path doesn't exist: {checkpoint_path}", "warn")
                        return False
            
            print_status("CHECKPOINT_PATH not configured in .env", "warn")
            return False
        else:
            print_status("CHECKPOINT_PATH not found in .env", "error")
            return False

def check_frontend_setup():
    """Check if frontend is set up"""
    if not Path('frontend/package.json').exists():
        print_status("frontend/package.json not found", "error")
        return False
    
    print_status("frontend/package.json exists", "ok")
    
    if Path('frontend/node_modules').exists():
        print_status("Frontend dependencies installed", "ok")
        return True
    else:
        print_status("Frontend dependencies not installed", "warn")
        print_status("Run: cd frontend && npm install", "info")
        return False

def check_checkpoint_dirs():
    """Check if any checkpoint directories exist"""
    checkpoint_dirs = list(Path('.').glob('PPO_*'))
    
    if checkpoint_dirs:
        print_status(f"Found {len(checkpoint_dirs)} checkpoint directories", "ok")
        for cp_dir in checkpoint_dirs[:3]:  # Show first 3
            print_status(f"  - {cp_dir}", "info")
        return True
    else:
        print_status("No checkpoint directories found (PPO_*)", "warn")
        print_status("You'll need a trained model checkpoint to use AI players", "info")
        return False

def main():
    """Main verification function"""
    print("\n" + "="*60)
    print("Hearts Game Setup Verification")
    print("="*60 + "\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Python Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Environment Configuration", check_env_file),
        ("Frontend Setup", check_frontend_setup),
        ("Checkpoint Directories", check_checkpoint_dirs),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 40)
        results.append(check_func())
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} checks")
    
    if passed == total:
        print_status("\n✓ All checks passed! You're ready to run the game.", "ok")
        print("\nNext steps:")
        print("  1. Run backend:  ./run_backend.sh")
        print("  2. Run frontend: ./run_frontend.sh")
        print("  3. Open http://localhost:3000")
    elif passed >= total - 1:
        print_status("\n⚠ Most checks passed. Review warnings above.", "warn")
        print("\nYou may still be able to run the game, but some features might not work.")
    else:
        print_status("\n✗ Some checks failed. Please fix the errors above.", "error")
        print("\nRefer to GAME_SETUP.md for detailed setup instructions.")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()


