# Multitask Learning

To run main.py, make sure that there is an existing directory named 'result_data/T_\<number of tasks\>', where \<number of tasks\> is the number of tasks you want for the current run of main.py. Then, you can run via:
  
  ```
  main.py --T <number of tasks> --N <number of instances> --d1 <d1> --d2 <d2> --d3 <d3>
  ```
  
  For T = 200 tasks, N = 1000, you can do:
  
  ```
  mkdir result_data/T_200
  main.py --T 200 --N 1000
  ```
  
