export MODEL_REPO=/opt/scratch/yongwu/gpt-fast/yongwww/llama-2-70b-chat-hf/model.pth
export DRAFT_MODEL_REPO=/opt/scratch/yongwu/gpt-fast/yongwww/llama-2-7b-chat-hf/model.pth

tp=4
/opt/bin/cuda-reserve.py --num-gpus $tp time torchrun --standalone --nproc_per_node=$tp generate.py \
  --compile  --draft_checkpoint_path $DRAFT_MODEL_REPO  --checkpoint_path $MODEL_REPO \
  --speculate_k 5 --prompt "def quicksort(arr):" \
  --max_new_tokens 200 --num_samples 50 --temperature 0 
  # top_p=0