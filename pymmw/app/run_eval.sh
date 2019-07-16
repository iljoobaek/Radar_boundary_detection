for file in DATA/ground_truth_*.txt; do
    python3 evaluation.py $file
done