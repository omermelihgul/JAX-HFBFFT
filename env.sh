module load PrgEnv-gnu/8.5.0
module load rocm/6.1.3 # may work with ROCm 6.0.0 and 6.2.x
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

conda create -n jax_env-frontier python=3.11.10 numpy scipy -c conda-forge
source activate jax_env-frontier

# Install jaxlib
pip install https://github.com/ROCm/jax/releases/download/rocm-jax-v0.4.35/jaxlib-0.4.35-cp311-cp311-manylinux_2_28_x86_64.whl

# Install the ROCm plugins
pip install https://github.com/ROCm/jax/releases/download/rocm-jax-v0.4.35/jax_rocm60_pjrt-0.4.35-py3-none-manylinux_2_28_x86_64.whl https://github.com/ROCm/jax/releases/download/rocm-jax-v0.4.35/jax_rocm60_plugin-0.4.35-cp311-cp311-manylinux_2_28_x86_64.whl

# Install JAX
pip install jax==0.4.35


