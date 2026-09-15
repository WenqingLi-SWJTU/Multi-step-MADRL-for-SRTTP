install environment according to torch-cpu.yaml
run trainer.py

The code in this repository currently uses the CPU version of PyTorch rather than the CUDA version. This was a deliberate choice for reproducibility: our experiments were run on a high-performance GPU (Nvidia GeForce  RTX 5090), and the CUDA/cuDNN libraries supported by those GPUs may not be compatible with the GPUs available to readers. To avoid such compatibility issues, we have uploaded the CPU build of PyTorch. If you have already installed the appropriate CUDA-related libraries, you can simply change default=False to default=True for the --cuda argument in get_config.py to enable GPU acceleration.
