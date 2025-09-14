export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

log_file="./local_weights/sft_with_query_8_distilling_seperate.log"
echo "Log file saved to: ${log_file}"
accelerate launch --config_file "scripts/zero3.yaml" \
--main_process_port 29500 \
src/video_r1/sft_with_query_seperate.py \
--config "src/video_r1/configs/sft_with_query_distilling_seperate_config.yaml" 2>&1 | tee "${log_file}"