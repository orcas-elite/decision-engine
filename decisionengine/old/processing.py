import pandas as pd
from pathlib import Path

def main():
    output_dir = Path('data')
    output_dir.mkdir(parents=True, exist_ok=True)

    summarize_data()

def summarize_data():
    hystrixConfigs = ["a1false-a2false-b1false-c1false","a1true-a2false-b1false-c1false","a1false-a2true-b1false-c1false", "a1false-a2false-b1true-c1false","a1false-a2false-b1false-c1true","a1true-a2true-b1false-c1false","a1true-a2false-b1true-c1false","a1true-a2false-b1false-c1true","a1false-a2true-b1true-c1false","a1false-a2true-b1false-c1true","a1false-a2false-b1true-c1true","a1true-a2true-b1true-c1false","a1true-a2true-b1false-c1true","a1true-a2false-b1true-c1true","a1false-a2true-b1true-c1true","a1true-a2true-b1true-c1true"]

    experimentDates = ['experiment-2018-09-16T15-13-59UTC','experiment-2018-09-19T19-10-21UTC','experiment-2018-09-28T15-05-51UTC','experiment-2018-10-09T18-32-17UTC','experiment-2018-10-21T11-28-15UTC','experiment-2018-11-04T08-32-02UTC','experiment-2018-11-18T10-44-58UTC','experiment-2018-12-02T07-32-48UTC']

    faultInjection=['b1-abort','b1-delay','c1-abort','c1-delay','c2-abort','c2-delay','d1-abort','d1-delay','e1-abort','e1-delay','e2-abort','e2-delay','nofault']

    # 16 hystrix configs, 13 fault injections
    total = 16 * 13 
    done = 0
    for config in hystrixConfigs:
        for fault in faultInjection:
            newFile = "data/" + config + "/" + fault + ".csv"
            output_dir = Path('data/' + config)
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for date in experimentDates:
                csvfile = "../experiments/" + date + "/" + config + "/" + fault + "/" + "response.csv"
                data = pd.read_csv(csvfile,usecols=[0,1,2,3,4,5])

                threshold = 0.065
                size = len(data.iloc[:,4])
                for i in range(0,size):
                    newRow = []
                    newRow.append(data.iloc[i,1])
                    newRow.append(data.iloc[i,4])
                    if (data.iloc[i,4] > threshold) or (data.iloc[i,1] == 500):
                        newRow.append(1)
                    else:
                        newRow.append(0)
                    rows.append(newRow)
            df = pd.DataFrame(rows,columns=['statusCode','responseTime','faultFound'])
            df.to_csv(newFile)
            done = done+1
            print(str((done/total)*100)+'%')

if __name__ == "__main__":
    main()