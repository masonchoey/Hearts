# PPO Self-Play Training Resume Functionality

## Overview

The `main_self_play.py` script now supports resuming training from previously saved PPO checkpoints. This allows you to:

- Continue interrupted training sessions
- Extend training beyond the original iteration limit
- Experiment with different hyperparameters on pre-trained models
- Save computational resources by building on previous work

## Features Added

### 1. Command Line Arguments

- `--resume <checkpoint_path>`: Resume from a specific checkpoint directory
- `--resume-from-latest <results_directory>`: Automatically find and resume from the latest checkpoint
- `--iterations <number>`: Set the number of training iterations (default: 250)

### 2. Automatic Checkpoint Discovery

The script can automatically:
- Find experiment directories within results folders
- Locate the latest checkpoint within an experiment
- Validate checkpoint integrity before resuming

### 3. Robust Path Handling

- Supports both relative and absolute paths
- Automatically converts paths to the correct format for Ray Tune
- Handles different checkpoint directory structures

## Usage Examples

### Basic Usage

```bash
# Start new training
python main_self_play.py

# Resume from latest checkpoint in results directory
python main_self_play.py --resume-from-latest PPO_2025-08-28_23-03-01

# Resume from specific checkpoint
python main_self_play.py --resume PPO_2025-08-28_23-03-01/PPO_hearts_env_self_play_*/checkpoint_000049
```

### Advanced Usage

```bash
# Resume with custom iteration count
python main_self_play.py --resume-from-latest PPO_2025-08-28_23-03-01 --iterations 500

# Resume for just a few more iterations
python main_self_play.py --resume-from-latest PPO_2025-08-28_23-03-01 --iterations 5

# Get help on all options
python main_self_play.py --help
```

## Directory Structure

The resume functionality expects the following directory structure:

```
PPO_2025-08-28_23-03-01/                    # Results directory
├── tuner.pkl                               # Required for resume
├── experiment_state-*.json                 # Experiment metadata
├── basic-variant-state-*.json              # Variant state
└── PPO_hearts_env_self_play_*/              # Experiment directory
    ├── checkpoint_000000/                   # Individual checkpoints
    │   ├── algorithm_state.pkl
    │   ├── policies/
    │   └── rllib_checkpoint.json
    ├── checkpoint_000049/                   # Latest checkpoint
    ├── params.json                          # Training parameters
    ├── progress.csv                         # Training progress
    └── result.json                          # Final results
```

## How It Works

### Checkpoint Discovery Process

1. **For `--resume-from-latest`**:
   - Scans the provided results directory
   - Finds experiment subdirectories matching `PPO_hearts_env_self_play_*`
   - Locates all checkpoint directories (`checkpoint_XXXXXX`)
   - Selects the checkpoint with the highest number

2. **For `--resume`**:
   - Uses the provided checkpoint path directly
   - Validates that the checkpoint exists and is valid

### Resume Process

1. **Path Resolution**: Converts checkpoint paths to the results directory path (required by Ray Tune)
2. **Validation**: Checks for the presence of `tuner.pkl` file
3. **Restoration**: Uses `tune.Tuner.restore()` to load the previous training state
4. **Continuation**: Continues training with the specified iteration limit

## Error Handling

The script provides clear error messages for common issues:

- **Missing checkpoint**: "Checkpoint path does not exist"
- **Invalid directory**: "No experiment directory found"
- **No checkpoints**: "No checkpoints found in experiment directory"
- **Missing tuner.pkl**: "Cannot find tuner.pkl in expected locations"

## Best Practices

### 1. Backup Important Checkpoints

```bash
# Create a backup before resuming
cp -r PPO_2025-08-28_23-03-01 PPO_2025-08-28_23-03-01_backup
```

### 2. Monitor Resource Usage

- Resumed training uses the same resource configuration as the original
- Ensure sufficient memory and CPU resources are available

### 3. Validate Results

- Check that training metrics continue smoothly from the resume point
- Monitor for any discontinuities in learning curves

## Troubleshooting

### Common Issues

1. **"URI has empty scheme" error**: 
   - Fixed by converting relative paths to absolute paths
   
2. **"tuner.pkl not found" error**:
   - Ensure you're pointing to the correct results directory
   - The directory should contain the `tuner.pkl` file

3. **Training immediately terminates**:
   - The model may have already reached the specified iteration limit
   - Increase the `--iterations` parameter

### Getting Help

```bash
# View all available options
python main_self_play.py --help

# Run the interactive example
python resume_training_example.py
```

## Implementation Details

### Key Functions Added

- `find_latest_checkpoint()`: Discovers the latest checkpoint in an experiment directory
- `find_experiment_checkpoint_dir()`: Locates experiment directories within results folders
- `parse_arguments()`: Handles command-line argument parsing
- `main()`: Wraps the training logic to prevent conflicts with Ray workers

### Ray Tune Integration

The resume functionality uses Ray Tune's built-in checkpoint restoration:

```python
# Restore from results directory
tuner = tune.Tuner.restore(
    restore_path, 
    trainable="PPO", 
    resume_unfinished=True, 
    restart_errored=True
)
```

## Testing

The implementation has been tested with:

- ✅ Resume from latest checkpoint
- ✅ Resume from specific checkpoint
- ✅ Custom iteration counts
- ✅ Path validation and error handling
- ✅ Integration with existing training pipeline

## Future Enhancements

Potential improvements could include:

- Support for resuming with modified hyperparameters
- Automatic checkpoint cleanup for disk space management
- Integration with experiment tracking systems
- Resume from specific iteration numbers rather than just latest
