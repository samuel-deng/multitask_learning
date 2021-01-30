#!/bin/sh

# Fix T, vary N
mkdir results

for i in 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 
    do
	mkdir results/N_$i
    done

for i in 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 
    do
	echo "################## N = $i ##################"
        python main.py --N $i --T 500 --d1 100 --d2 50 --d3 50 --r 10 --eta 1 --output_dir "results/N_$i"
    done
