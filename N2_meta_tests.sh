#!/bin/sh

# Make the meta test result directory
mkdir meta_test_results
mkdir meta_test_results/trial$1

# Main Loop (10 trials)
echo "########### TRIAL = $1 ##############"
for N2 in 50 75 100 125 150 175 200 225 250
    do
	echo "########## N = $N2 (trial = $1) ##########"
	# python meta_test.py --N 2000 --T 100 --N2 $N2 --d1 100 --d2 50 --d3 50 --r 10 --eta 1 --output_dir "meta_test_results/N2_$N2/"
	python meta_test.py --output_dir meta_test_results/trial$1/ --A_and_task_dir meta_test_results/persistent/  --eta 1 --N 2000 --T 100 --N2 $N2 --d1 100 --d2 50 --d3 50 --r 10 --seed $1
    done
