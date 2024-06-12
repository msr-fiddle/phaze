mbs=$1
#mode = run_solver # prepopulate_estimates, extract_graph

# if the model is to be extracted from scratch, then the following flag should be added
#   --force_reextract_model True
#   This mode requires space to extract and execute the forward pass of the model
# else the model from GraphExtractor/out folder will be used.

cd ..

if [ ! -d "GraphExtractor/out/GPT" ]; then
  mkdir GraphExtractor/out/GPT
fi

# GPT2
echo "mbs: $mbs";
python3 phaze.py \
        --phaze_model gpt2 \
        --phaze_exec_type run_solver \
        --phaze_micro_batch_size $mbs \
        --phaze_sequence_length 1024 \
        --phaze_hbm_size 32 64 80
