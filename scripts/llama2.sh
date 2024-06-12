mbs=$1
mode = run_solver # prepopulate_estimates, extract_graph

# if the model is to be extracted from scratch, then the following flag should be added
#   --force_reextract_model True
#   This mode requires space to extract and execute the forward pass of the model
# else the model from GraphExtractor/out folder will be used.

cd ..

if [ ! -d "GraphExtractor/out/Llama" ]; then
  mkdir GraphExtractor/out/Llama
fi

echo "mbs: $mbs";
python3 phaze.py \
        --phaze_model llama2 \
        --phaze_exec_type run_solver \
        --phaze_micro_batch_size $mbs \
        --phaze_sequence_length 4096 \
        --phaze_hbm_size 32 64 80 
