#!/bin/sh

# Make the meta test result directory
mkdir meta_test_results

for T in 10 20 30 40 50 
    do
	rm -rf meta_test_results/T_$T
	mkdir meta_test_results/T_$T
    done

for T in 10 20 30 40 50 
    do
	echo "########## T = $T ##########"
	python meta_test.py --N 2000 --T $T --N2 100 --d1 100 --d2 50 --d3 50 --r 10 --eta 1 --output_dir "meta_test_results/T_$T/"
    done
