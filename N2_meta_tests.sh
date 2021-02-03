#!/bin/sh

# Make the meta test result directory
mkdir meta_test_results

for N2 in 50 75 100 125 150 175 200 225 250
    do
	rm -rf meta_test_results/N2_$N2
	mkdir meta_test_results/N2_$N2
    done

for N2 in 50 75 100 125 150 175 200 225 250
    do
	echo "########## N = $N2 ##########"
	python meta_test.py --N 2000 --T 100 --N2 $N2 --d1 100 --d2 50 --d3 50 --r 10 --eta 1 --output_dir "meta_test_results/N2_$N2/"
    done
