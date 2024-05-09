method1=direct
datasets1=(
    openreview__rand__gpt4__bullet openreview__rand__gpt4__bullet__cond
    openreview__rand__gpt4__prediction openreview__rand__gpt4__prediction__cond
    yelp__restaurant__gpt35__bullet yelp__restaurant__gpt35__bullet__cond
    yelp__restaurant__gpt35__prediction
    openreview__rand__gpt4__predictionllama yelp__restaurant__gpt35__predictionllama
)

method3=baseline
datasets3=(
    openreview__rl
    openreview__rand
    yelp__restaurant
)

for dataset in "${datasets3[@]}";
do
    for experiment in same_diff_item
    do
        python3 roc.py \
            --method $method3 \
            --dataset $dataset \
            --experiment $experiment
    done
done

for dataset in "${datasets1[@]}";
do
    for experiment in same_diff_item oneside_degrade_same llm_gen_review bad_llm_review
    do
        python3 roc.py \
            --method $method1 \
            --dataset $dataset \
            --experiment $experiment
    done
done