
python toy_sae/run_sweep.py \
    --sweep_name "bigger-grid-sweep" \
    --sweep_count 210 \
    --sweep_method grid \
    --n_dims 256 \
    --n_surplus 256 \
    --n_examples 16384 \
    --sparse_fraction 0.01 \
    --n_hidden 512 \
    --seed 0