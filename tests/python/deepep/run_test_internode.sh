# set your master node ip
RANK0_IP="80.48.29.109"
IP=$(hostname -I | awk '{print $1}')

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir" || exit

export WORLD_SIZE=2
export HCCL_BUFFSIZE=3000
export HCCL_LOGIC_SUPERPOD_ID=0
export MASTER_ADDR=${RANK0_IP}
if [ "${IP}" == "${RANK0_IP}" ]; then
  echo "env rank 0"
  export RANK=0
else
  echo "env rank 1"
  export RANK=1
fi

python test_low_latency.py
