python ./VisualizeCodebook.py \
  --config=./configs/codebook.yml \
  --no_cuda 1 \
  --gpu 0 \
  --code_path "./output/result.npz" \
  --VQVAE_model_path "../../pretrained_model/codebook_checkpoint_best.bin" \
  --stage inference
