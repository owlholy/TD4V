if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

now=$(date +"%Y%m%d_%H%M%S")
python -m torch.distributed.run --master_port 2339 --nproc_per_node=4 \
         train_vision.py  --config configs/hmdb51.yaml --log_time $now

