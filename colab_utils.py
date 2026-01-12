#!/usr/bin/env python3
"""
Google Colab Utilities for Rolling Training Pool

Provides utilities for:
- Detecting Colab environment
- Mounting Google Drive
- Managing Drive paths
- Handling Colab-specific configurations
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple


def is_colab() -> bool:
    """
    Detect if running in Google Colab environment.
    
    Returns:
        True if running in Colab, False otherwise
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """
    Mount Google Drive in Colab.
    
    Args:
        mount_point: Where to mount Drive (default: /content/drive)
        
    Returns:
        True if successful, False otherwise
    """
    if not is_colab():
        print("⚠️  Not running in Colab, skipping Drive mount")
        return False
    
    try:
        from google.colab import drive
        
        # Check if already mounted
        if os.path.exists(os.path.join(mount_point, "MyDrive")):
            print(f"✅ Google Drive already mounted at {mount_point}")
            return True
        
        # Mount Drive
        print(f"📂 Mounting Google Drive at {mount_point}...")
        drive.mount(mount_point, force_remount=False)
        print(f"✅ Google Drive mounted successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        return False


def get_drive_path(relative_path: str = "Hearts_RL") -> Path:
    """
    Get the path to a directory in Google Drive.
    
    Args:
        relative_path: Path relative to MyDrive (default: Hearts_RL)
        
    Returns:
        Path object pointing to the Drive location
    """
    if is_colab():
        drive_base = Path("/content/drive/MyDrive")
    else:
        # For local development, use current directory
        drive_base = Path.cwd()
    
    return drive_base / relative_path


def setup_colab_environment(
    project_name: str = "Hearts_RL",
    mount_drive: bool = True
) -> Tuple[Path, Path]:
    """
    Set up the Colab environment for training.
    
    Args:
        project_name: Name of the project folder in Drive
        mount_drive: Whether to mount Google Drive
        
    Returns:
        Tuple of (project_path, pool_path)
    """
    print("=" * 80)
    print("SETTING UP GOOGLE COLAB ENVIRONMENT")
    print("=" * 80)
    
    # Detect environment
    if is_colab():
        print("✅ Running in Google Colab")
    else:
        print("ℹ️  Running locally (not Colab)")
    
    # Mount Drive if in Colab
    if is_colab() and mount_drive:
        if not mount_google_drive():
            raise RuntimeError("Failed to mount Google Drive")
    
    # Set up project paths
    project_path = get_drive_path(project_name)
    pool_path = project_path / "models" / "pool"
    
    # Create directories if they don't exist
    project_path.mkdir(parents=True, exist_ok=True)
    pool_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Project Paths:")
    print(f"   Project: {project_path}")
    print(f"   Pool: {pool_path}")
    print(f"   Exists: {project_path.exists()}")
    
    # Check available space
    if is_colab():
        import shutil
        total, used, free = shutil.disk_usage(project_path)
        print(f"\n💾 Google Drive Space:")
        print(f"   Total: {total // (2**30)} GB")
        print(f"   Used: {used // (2**30)} GB")
        print(f"   Free: {free // (2**30)} GB")
        
        # Warn if low on space
        if free < 5 * (2**30):  # Less than 5GB
            print(f"   ⚠️  WARNING: Low on Drive space! Consider freeing up space.")
    
    print("=" * 80)
    
    return project_path, pool_path


def install_colab_dependencies():
    """
    Install required packages for Colab that aren't pre-installed.
    """
    if not is_colab():
        print("ℹ️  Not in Colab, skipping dependency installation")
        return
    
    print("=" * 80)
    print("INSTALLING COLAB DEPENDENCIES")
    print("=" * 80)
    
    # List of packages to install
    packages = [
        "ray[rllib]==2.48.0",
        "open_spiel==1.6",
        "wandb",
        "gymnasium",
    ]
    
    import subprocess
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", package
            ])
            print(f"   ✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Failed to install {package}: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Dependency installation complete")
    print("=" * 80)


def sync_code_to_drive(
    source_files: list,
    project_path: Path
):
    """
    Copy Python code files to Google Drive for persistence.
    
    Args:
        source_files: List of Python files to copy
        project_path: Destination path in Drive
    """
    import shutil
    
    print(f"\n📋 Syncing code files to Drive...")
    
    for file in source_files:
        src = Path(file)
        if src.exists():
            dst = project_path / src.name
            shutil.copy2(src, dst)
            print(f"   ✅ Copied {src.name}")
        else:
            print(f"   ⚠️  File not found: {file}")
    
    print(f"✅ Code sync complete")


def get_checkpoint_storage_path(project_path: Path) -> Path:
    """
    Get the path for storing Ray checkpoints in Drive.
    
    Args:
        project_path: Project path in Drive
        
    Returns:
        Path for checkpoint storage
    """
    checkpoint_path = project_path / "ray_results"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def check_colab_resources():
    """
    Check available Colab resources (GPU, RAM, disk).
    """
    print("=" * 80)
    print("COLAB RESOURCES")
    print("=" * 80)
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (2**30)
            print(f"🎮 GPU: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
        else:
            print("⚠️  No GPU available")
    except ImportError:
        print("ℹ️  PyTorch not installed, cannot check GPU")
    
    # Check RAM
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"\n💾 RAM:")
        print(f"   Total: {ram.total / (2**30):.1f} GB")
        print(f"   Available: {ram.available / (2**30):.1f} GB")
    except ImportError:
        print("ℹ️  psutil not installed, cannot check RAM")
    
    # Check CPU
    import os
    cpu_count = os.cpu_count()
    print(f"\n🖥️  CPU Cores: {cpu_count}")
    
    print("=" * 80)


def create_colab_checkpoint_config(
    project_path: Path,
    checkpoint_frequency: int = 10,
    num_to_keep: int = 3
):
    """
    Create a checkpoint config suitable for Colab with Drive storage.
    
    Args:
        project_path: Project path in Drive
        checkpoint_frequency: Save checkpoint every N iterations
        num_to_keep: Number of checkpoints to keep in Ray results
        
    Returns:
        Dictionary with checkpoint configuration
    """
    checkpoint_path = get_checkpoint_storage_path(project_path)
    
    return {
        "checkpoint_frequency": checkpoint_frequency,
        "num_to_keep": num_to_keep,
        "checkpoint_at_end": True,
        "local_dir": str(checkpoint_path)
    }


def get_wandb_config_for_colab(project_name: str = "hearts-ppo-colab-rolling-pool"):
    """
    Get W&B configuration for Colab.
    
    Args:
        project_name: W&B project name
        
    Returns:
        Dictionary with W&B configuration
    """
    return {
        "project": project_name,
        "name": f"colab-rolling-pool",
        "tags": ["colab", "rolling-pool", "hearts"],
        "notes": "Training with rolling checkpoint pool on Google Colab"
    }


def save_training_state(
    project_path: Path,
    iteration: int,
    metrics: dict
):
    """
    Save training state to Drive for resume capability.
    
    Args:
        project_path: Project path in Drive
        iteration: Current training iteration
        metrics: Current metrics dictionary
    """
    import json
    from datetime import datetime
    
    state_file = project_path / "training_state.json"
    
    state = {
        "last_iteration": iteration,
        "last_update": datetime.now().isoformat(),
        "metrics": metrics,
    }
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"💾 Saved training state to Drive (iteration {iteration})")


def load_training_state(project_path: Path) -> dict:
    """
    Load previous training state from Drive.
    
    Args:
        project_path: Project path in Drive
        
    Returns:
        Dictionary with training state or empty dict if not found
    """
    import json
    
    state_file = project_path / "training_state.json"
    
    if state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
        print(f"✅ Loaded training state from Drive")
        print(f"   Last iteration: {state.get('last_iteration', 'unknown')}")
        print(f"   Last update: {state.get('last_update', 'unknown')}")
        return state
    else:
        print("ℹ️  No previous training state found")
        return {}


def main():
    """Demo of Colab utilities."""
    print("=" * 80)
    print("GOOGLE COLAB UTILITIES DEMO")
    print("=" * 80)
    
    # Check if in Colab
    if is_colab():
        print("✅ Running in Google Colab")
    else:
        print("ℹ️  Not running in Colab (local environment)")
    
    # Check resources
    check_colab_resources()
    
    # Set up environment (without actually mounting in demo)
    print("\n📁 Example paths:")
    project_path = get_drive_path("Hearts_RL")
    pool_path = project_path / "models" / "pool"
    print(f"   Project: {project_path}")
    print(f"   Pool: {pool_path}")
    
    print("\n" + "=" * 80)
    print("Ready for Colab training!")
    print("=" * 80)


if __name__ == "__main__":
    main()

