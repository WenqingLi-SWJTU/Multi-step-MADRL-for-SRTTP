Install environment according to torch-cpu.yaml

Run trainer.py

The code in this repository currently uses the CPU version of PyTorch rather than the CUDA version. This was a deliberate choice for reproducibility: our experiments were run on a Nvidia GeForce RTX 5090 GPU, and the libraries supported by the GPU may not be compatible with the GPUs of readers. To avoid such compatibility issues, we uploaded the CPU build of PyTorch. If you have already installed the appropriate CUDA-related libraries, you can simply change default=False to default=True for the --cuda argument in get_config.py to enable GPU acceleration.
