source .venv/bin/activate

CMD_PREFIX="${1:-""}"

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

export head_node=${nodes_array[0]}
if [[ "$head_node" == *"jrc"* ]]; then
    head_node_i="${head_node}i"
else
    head_node_i="$head_node"
fi
# echo "Head node: $head_node, head node with i: $head_node_i"
head_node_ip=$(nslookup $head_node_i | grep 'Address' | tail -n1 | awk '{print $2}')

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

export RAY_memory_usage_threshold=0.99

#MEM_AVAIL=$(srun --mpi=none --nodes=1 --ntasks=1 -w "$head_node" free -g | awk '/^Mem:/ {print $7}')

RAY_OBJ_STORE=$((16 * 1024**3))

TEMP_DIR=$RAY_TMPDIR
TEMP_DIR_HEAD="$TEMP_DIR"
# rm -rf $TEMP_DIR_HEAD/*
# mkdir -p $TEMP_DIR_HEAD

# NOTE: do NOT add `--temp-dir` on a shared FS here. Ray only honors
# --temp-dir on the head and forces that same path on EVERY node's raylet;
# a shared /gpfs temp-dir overloads the GCS -> raylets time out at startup.
# Ray's temp-dir must be node-local (default /tmp/ray).

echo "Starting HEAD at $head_node, RAY_TMPDIR: $RAY_TMPDIR, RAY_OBJ_STORE: $RAY_OBJ_STORE, RAY_SPILL_DIR: $RAY_SPILL_DIR"
srun --export=ALL,VLLM_HOST_IP="$head_node_ip" --nodes=1  --ntasks=1 --gres=gpu:4  --overlap -w "$head_node" \
    $CMD_PREFIX ray start --head --node-ip-address="$head_node_ip" --include-dashboard=0 --disable-usage-stats --object-store-memory=$RAY_OBJ_STORE --port=$port \
    --num-gpus 4  --block &

until ray status --address "$ip_head" >/dev/null 2>&1; do
  # echo "Ray cluster is still starting up..."
  sleep 5
done


# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    # Every 4 nodes, add a vllm_instance resource
    slurm_node=${nodes_array[$i]}
    ray_node="$slurm_node"
    if [[ "$ray_node" == *"jrc"* ]]; then
        ray_node="${ray_node}i"
    fi
    node_i_ip_addr=$(nslookup $ray_node | grep "Address" | tail -n1 | awk "{print \$2}")
    echo "Starting WORKER $i at $slurm_node with Ray node address $ray_node"
    srun --export=ALL,VLLM_HOST_IP="$node_i_ip_addr" --nodes=1 --ntasks=1 --gres=gpu:4 -w "$slurm_node" \
         $CMD_PREFIX ray start --address "$ip_head"  --node-ip-address="$node_i_ip_addr" \
         --num-gpus 4  --block &
    sleep 5
done


export RAY_ADDRESS=$ip_head
echo "$RAY_ADDRESS" > ray_address.txt
# sleep 10
