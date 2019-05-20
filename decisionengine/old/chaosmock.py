import pandas as pd

def get_result(service,operation,action,hystrix):
    df = pd.read_csv("/Users/niklaskhf/GoogleDrive/Uni/BScSWT/Bachelor/code/decisionengine/data/" + hystrix + "/" + operation + "-" + action + ".csv")
    result = df.sample(1)
    print(result)
    return str(result.iloc[0,3])

def print_experiment(service,operation,action,hystrix):
    print("Running Experiment: " + service + "-" + operation + "-" + action + " for hystrix configuration: " + hystrix)