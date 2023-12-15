SEEDS=(7777 1225)
MINERS=("no_miner" "batch_easy_hard_miner")
ACTIVATIONS=("no" "relu" "soft_plus" "leaky_relu" "hard_swish" "selu" "celu" "gelu" "silu" "mish")

for miner in "${MINERS[@]}"
do
    for seed in "${SEEDS[@]}"
    do
        for activation in "${ACTIVATIONS[@]}"
        do
            echo "Start ${miner} with seed ${seed} and activation ${activation}"
            python -m src.${miner} --seed ${seed} --trial ${miner}_seed_${seed}_${activation} --activation ${activation}
            echo "End ${miner} with seed ${seed} and activation ${activation}"
        done
    done
done
