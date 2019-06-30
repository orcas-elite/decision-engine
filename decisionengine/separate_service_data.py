import pandas as pd 
from pathlib import Path
import os 


experimentDates = ['experiment-2018-09-16T15-13-59UTC','experiment-2018-09-19T19-10-21UTC','experiment-2018-09-28T15-05-51UTC','experiment-2018-10-09T18-32-17UTC','experiment-2018-10-21T11-28-15UTC','experiment-2018-11-04T08-32-02UTC','experiment-2018-11-18T10-44-58UTC','experiment-2018-12-02T07-32-48UTC']
faultInjection=['b1-abort','b1-delay','c1-abort','c1-delay','c2-abort','c2-delay','d1-abort','d1-delay','e1-abort','e1-delay','e2-abort','e2-delay','nofault']
hystrixConfigs = ["a1false-a2false-b1false-c1false","a1true-a2false-b1false-c1false","a1false-a2true-b1false-c1false", "a1false-a2false-b1true-c1false","a1false-a2false-b1false-c1true","a1true-a2true-b1false-c1false","a1true-a2false-b1true-c1false","a1true-a2false-b1false-c1true","a1false-a2true-b1true-c1false","a1false-a2true-b1false-c1true","a1false-a2false-b1true-c1true","a1true-a2true-b1true-c1false","a1true-a2true-b1false-c1true","a1true-a2false-b1true-c1true","a1false-a2true-b1true-c1true","a1true-a2true-b1true-c1true"]

new_dir = 'data_service'

done = 0
total = 8*16*13
for date in experimentDates:
        for config in hystrixConfigs:
            for fault in faultInjection:
                a1file = new_dir + "/" + "a1/" + date + "/" + config + "/" + fault + ".csv"
                a2file = new_dir + '/' + 'a2/' + date + "/" + config + '/' + fault + '.csv'

                a1dir = os.path.dirname(a1file)
                a2dir = os.path.dirname(a2file)

                for directory in [a1dir,a2dir]:
                    if not os.path.exists(directory):
                        os.makedirs(directory)


                csvfile = "experiments/" + date + "/" + config + "/" + fault + "/" + "response.csv"
                data = pd.read_csv(csvfile,usecols=[0,1,2,3,4,5])

                a1rows = []
                a2rows = []

                size = len(data.iloc[:,4])
                for i in range(0,size):
                    newRow = []
                    newRow.append(data.iloc[i,1])
                    newRow.append(data.iloc[i,4])
                    mes = data.iloc[i,3]
                    if "a1" in mes:
                        a1rows.append(newRow)
                    elif "a2" in mes:
                        a2rows.append(newRow)
                    else: 
                        continue

                cols = ['statusCode', 'responseTime']
                a1data = pd.DataFrame(a1rows, columns=cols)
                a2data = pd.DataFrame(a2rows, columns=cols)
                a1data.to_csv(a1file)
                a2data.to_csv(a2file)
                done = done+1
                print(str((done/total)*100)+'%')