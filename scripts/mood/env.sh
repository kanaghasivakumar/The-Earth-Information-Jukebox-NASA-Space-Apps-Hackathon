# Source me: the Magenta RT env (separate from the shared jukebox env).
export MRT_ENV=/projects/e32706/omb8654/conda/envs/mrt
export PATH=$MRT_ENV/bin:$PATH
export HF_HOME=/projects/e32706/omb8654/hf_cache
export MAGENTA_HOME=/projects/e32706/omb8654/magenta_home
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
cd /projects/e32706/omb8654/jukebox-wt/magenta
