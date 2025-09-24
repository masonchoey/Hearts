# NVIDIA T4 GPU Setup Guide for Hearts RL Training

## T4 GPU Specifications
- **Memory**: 16 GB GDDR6
- **CUDA Cores**: 2,560
- **Tensor Cores**: 320 (for mixed precision)
- **Performance**: 65 TFLOPS FP16, 8.1 TFLOPS FP32
- **Power**: 70W TDP (very efficient!)

## Prerequisites

### 1. Install NVIDIA Drivers
```bash
# Check if drivers are installed
nvidia-smi

# If not installed, install latest drivers
sudo apt update
sudo apt install nvidia-driver-535  # or latest version
```

### 2. Install CUDA Toolkit
```bash
# Install CUDA 11.8 (compatible with most PyTorch versions)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-11-8
```

### 3. Install PyTorch with CUDA Support
```bash
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Verify GPU Setup
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## T4-Optimized Training Configuration

### Key Optimizations for T4:

1. **Large Batch Sizes**: Leverage 16GB memory
   - `train_batch_size=32000` (vs 12000 in original)
   - `minibatch_size=128` (vs 32 in original)

2. **Deeper Networks**: Utilize CUDA cores
   - `fcnet_hiddens=[1024, 1024, 512]` (3-layer vs 2-layer)

3. **More Training Epochs**: Maximize GPU utilization
   - `num_epochs=30` (vs 20 in original)

4. **Mixed Precision**: Use Tensor cores for speed
   - Enable with `--use-mixed-precision` flag

### Running T4-Optimized Training

```bash
# Basic T4 training
python main_self_play_optimized.py --iterations 500

# With mixed precision (uses Tensor cores)
python main_self_play_optimized.py --iterations 500 --use-mixed-precision

# Resume from checkpoint
python main_self_play_optimized.py --resume-from-latest PPO_results/ --iterations 1000
```

## Monitoring GPU Usage

### Real-time Monitoring
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or use htop-like interface
nvtop  # install with: sudo apt install nvtop
```

### Expected T4 Utilization
- **GPU Utilization**: 80-95% during training
- **Memory Usage**: 8-12 GB out of 16 GB
- **Power Draw**: 60-70W
- **Temperature**: 65-75°C

## Performance Expectations

### Training Speed Improvements with T4:
- **~5-10x faster** than CPU-only training
- **~2-3x larger batch sizes** possible
- **~50% faster convergence** due to larger batches

### Typical Training Times:
- **500 iterations**: ~2-4 hours (vs 10-20 hours on CPU)
- **1000 iterations**: ~4-8 hours
- **Full training**: ~8-16 hours for high-quality agent

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce batch sizes in the config:
   ```python
   train_batch_size=16000,  # Reduce from 32000
   minibatch_size=64,       # Reduce from 128
   ```

2. **GPU Not Detected**
   ```
   CUDA available: False
   ```
   **Solution**: Check driver installation and PyTorch CUDA version

3. **Low GPU Utilization (<50%)**
   - Increase batch sizes
   - Reduce `num_env_runners` if CPU-bound
   - Enable mixed precision

4. **Mixed Precision Errors**
   ```
   RuntimeError: Expected tensor for argument #1 'input' to have scalar type Half
   ```
   **Solution**: Disable mixed precision temporarily and update PyTorch

## Advanced T4 Optimizations

### 1. Multi-GPU Setup (if you have multiple T4s)
```python
.resources(
    num_gpus=2,  # Use 2 T4 GPUs
    num_gpus_per_learner=2,
)
```

### 2. CPU-GPU Balance
```python
# Adjust based on your CPU count
.env_runners(
    num_env_runners=4,      # Scale with available CPUs
    num_envs_per_env_runner=2,
)
```

### 3. Memory-Optimized Training
```python
# For maximum memory efficiency
.training(
    train_batch_size=48000,    # Push T4 to its limits
    minibatch_size=256,        # Large minibatches
    num_epochs=40,             # More learning per batch
)
```

## Cost Optimization

### Cloud T4 Pricing (approximate):
- **Google Cloud**: $0.35/hour
- **AWS**: $0.526/hour  
- **Azure**: $0.45/hour

### Training Cost Estimates:
- **500 iterations**: $1-3
- **Full training**: $3-8
- **Experimentation**: $10-20/day

### Cost-Saving Tips:
1. Use preemptible/spot instances (50-70% discount)
2. Stop training during evaluation phases
3. Use checkpointing to resume interrupted training
4. Monitor and stop underperforming runs early

## Next Steps

1. Run the optimized configuration: `python main_self_play_optimized.py`
2. Monitor GPU utilization with `nvidia-smi`
3. Experiment with batch sizes based on your specific setup
4. Enable mixed precision for maximum speed
5. Scale up to longer training runs (1000+ iterations)

The T4 is an excellent choice for RL training - it provides great performance per dollar and its 16GB memory allows for large batch sizes that can significantly improve training quality!
