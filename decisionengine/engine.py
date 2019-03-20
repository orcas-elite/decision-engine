import csv 
import random
import pandas as pd
import randomSelect
import experiment 
import architecture 
import strategy
import json
import os
import sys
import chaosmock
import bayesian_low 

def main():
    # add this directory's modules to sys path
    os.system("cp chaosmock.py /usr/local/lib/python3.7/site-packages/chaosmock.py")

    # init architecture from model
    arch = architecture.new('architecture_model.json')
    
    # randomly select experiments and count in csv
    #randomStrategy = randomSelect.RandomSelection()

    #context = strategy.Context(randomStrategy)
    
    context = strategy.Context(bayesian_low.BayesianLow())
    
    context.init(arch)
    print("Starting selection")

    # Ignore this for now.
    #configs = ["a1false-a2false-b1false-c1false","a1true-a2false-b1false-c1false","a1false-a2true-b1false-c1false", "a1false-a2false-b1true-c1false","a1false-a2false-b1false-c1true","a1true-a2true-b1false-c1false","a1true-a2false-b1true-c1false","a1true-a2false-b1false-c1true","a1false-a2true-b1true-c1false","a1false-a2trueb1false-c1true","a1false-a2false-b1true-c1true","a1true-a2true-b1true-c1false","a1true-a2true-b1false-c1true","a1true-a2false-b1true-c1true","a1false-a2true-b1true-c1true","a1true-a2true-b1true-c1true"]
    ## TODO: Add hystrix iterations
    for i in range(0,100):
        exp = context.next_experiment()
        args = exp.ssh.probes[0].provider.arguments
        result = chaosmock.get_result(args[0],args[1],args[2],args[3])

        context.process_result(exp,result)
    print("Finished selection")


    """while i < 100:
        exp = context.next_experiment()
        with open('experiment.json', 'w') as f:
            f.write(json.dumps(exp.reprJSON(),indent=4))
        os.system("chaos run experiment.json")

        with open('journal.json') as f:
            data = json.load(f)
            if (data["steady_states"]["after"] != None) and (data["steady_states"]["after"]["steady_state_met"]==True):
                result = '0'
            else:
                result = '1'

        context.process_result(exp,result)
        i = i + 1
    print("Finished selection")
    """


if __name__ == "__main__":
    main()