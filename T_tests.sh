#!/bin/sh

# Fix N, vary T
mkdir results

for i in 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500
    do
	mkdir results/T_$i
    done

for i in 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500
    do
	echo "################## T = $i ##################"
        python main.py --T $i --N 5000 --d1 100 --d2 50 --d3 50 --r 10 --output_dir "results/T_$i/"
    done
