#!/bin/bash

# Run experiments with different optimization modes
# python Bloop.py mode=dbgd auxiliary_loss=lipschitz full_batch=true
# python Bloop.py mode=standard auxiliary_loss=lipschitz full_batch=true
# python Bloop.py mode=baseline auxiliary_loss=lipschitz full_batch=true optimization.baseline.aux_weight=0.5
# python plot_optimization_modes.py


# python Bloop.py mode=dbgd auxiliary_loss=lipschitz full_batch=true
# python Bloop.py mode=dbgd auxiliary_loss=l2_norm full_batch=true
# python Bloop.py mode=dbgd auxiliary_loss=test_loss full_batch=true

# # Create comparative plots
# python plot_comparison.py

python Bloop.py mode=dbgd auxiliary_loss=lipschitz full_batch=true &
python Bloop.py mode=baseline auxiliary_loss=lipschitz full_batch=true optimization.baseline.aux_weight=0.01 &
python Bloop.py mode=baseline auxiliary_loss=lipschitz full_batch=true optimization.baseline.aux_weight=0.1 &
python Bloop.py mode=baseline auxiliary_loss=lipschitz full_batch=true optimization.baseline.aux_weight=1 &
python Bloop.py mode=baseline auxiliary_loss=lipschitz full_batch=true optimization.baseline.aux_weight=10 &

# Wait for all background processes to complete
wait

# Run the plotting script after all experiments are done
python plot_baseline_comparison.py