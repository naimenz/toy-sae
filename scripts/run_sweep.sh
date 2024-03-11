
python toy_sae/run_sweep.py \
    --sweep_name "grid-sweep-1" \
    --sweep_count 100 \
    --sweep_method grid \
    --n_dims 10 \
    --n_surplus 5 \
    --n_examples 10000 \
    --sparse_fraction 0.1 \
    --n_hidden 15 \
    --seed 0