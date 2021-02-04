# Multitask Learning

## Meta-Test (N2 Experiments)
To run the meta-test for values of N2, do the following. Before you begin, make sure you already have a directory named 'meta_test_results' with a 'persistent' subdirectory. If not:

    ```
    mkdir meta_test_results
    mkdir meta_test_results/persistent
    ```

First, run ONCE (throughout all 10 trials):
    ```
    python pre_meta_test.py
    ```

This will generate the underlying A system tensor throughout the 10 trials, as well as Y0 and Z0. Then, to run trial `k`, you should simply run the bash script:

    ```
    ./N2_meta_tests.sh k
    ```

where the command line argument `k` is the current number trial. Running this will perform the training and Meta-Test procedure for the desired values of N2:

    ```
    50 75 100 125 150 175 200 225 250
    ```

## Meta-Test (T Experiments)
To run the expeirments for values of T, do the following. Before you begin, make sure you already have a directory named `meta_test_results` with a `persistent` subdirectory. If not:

```
mkdir meta_test_results
mkdir meta_test_results/persistent
```

First, run ONCE (throughout all 10 trials):
    ```
    python pre_meta_test_.py
    ```
This generates the underlying A system tensor throughout the 10 trials, as well as Y0 and Z0. To run trial `k`, simply run the bash script with `k` as first argument:

    ```
    ./T_meta_tests.sh k
    ```

where the command line argument `k` is the current number trial. Running this will perform the training and Meta-Test procedure for the values of T:

```
10 20 30 40 50 60 70 80 90 100
```
  
