# States
- In paper: duration_group, time_group, last 4 execution results 
- For chaos: #instances-service, #incoming-dependencies, #operations-service, circuitBreaker, last 4 execution results 

# Actions
- In paper: prioritization
- For chaos: ?

# Reward
- In paper: Various, e.g. detected_failures
- For chaos: 1/0





# Sequene
- Run experiments in sequences
- Run 80% of experiments in each episode



# Experiment Runs
- experiment_results_1/2/5.csv: Irrelevant first runs, parameters cb/dependency count/op per instance
- experiment_results_double_01/02/04/05: 25000 experiment run of parameters cb and dependency count
- experiment_results_double_10: 600000 experiment run of parameters cb and dependency count
- experiment_results_double_11: 25 * 25000 experiment run of parameters cb and dependency count
- experiment_results_double_history_01: 600000 experiment run of parameters cb/dependency count/last 4 history results
- experiment_results_double_history_02: 25 * 25000 experiment run of parameters cb/dependency count/last 4 history results