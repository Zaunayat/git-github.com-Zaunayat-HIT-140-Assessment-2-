import pandas as pd
import numpy as np
import scipy.stats as st
import math
import statistics as stats
import matplotlib.pyplot as plt

# Push the data file as numpy 

p = np.genfromtxt('po1_data.txt', delimiter=',')

print(p)

#reading the data file as panda data framework 

df = pd.DataFrame(p)

print(df)

#describing the summary of the data set

df.describe()

print(df.dtypes)

#adding the column in the data from the instruction 

df.info()

df.columns = ['subject identifier', ' jitter in %', 'absolt jitter in ms', 'jitter as r.a.p.', 'jitter as p.p.q.5','jitter as d.d.p.' ,
              'shimmer in %','absolt shimmer in dB', 'shimmer as a.p.q.3', 'shimmer as a.p.q.5','shimmer as a.p.q.11','shimmer as d.d.a', 
              'autocorrelation btwn NHR and HNR', 'NHR','HNR','median pitch','mean pitch','stdv of pitch','min pitch',
              'max pitch','number of pulses','number of periods','mean period','stdv of period','fraction of unvoiced frames',
              'nmbr of voice breaks','degree of voice breaks','UPDRS','PD indicator']

print(df)

column_names = list(df.columns)
df.isnull()

#PD Indicator has null value so we disect the data set into two different sample size 

df1 = df[df["PD indicator"] == 1]
df0 = df[df["PD indicator"] == 0]


print(df1)
print(df0)

#we i in range(1,28) is loop that iterates the numbers 1 to 27. it will analyse the column 1 to 17

variable_analysis = []
for i in range(1,28):
    print('analysis of', column_names[i])
    print()
    sample1 = df1.iloc[:, i].values
    sample2 = df0.iloc[:, i].values
    
    

    # Mean, Standard Deviation, Sample Size (n) of Sample 1
    
    
    x_bar1 = np.mean(sample1)
    stdv1 = np.std(sample1)
    n1 = np.size(sample1)
    print("\n Analysis of sample 1: Mean = %.2f, Std. Dev. = %.2f, n = %d" % (x_bar1, stdv1, n1))
   
   
   # Histogrtam of all the Variables of Sample 1 
    
    plt.hist(sample1, color='blue', edgecolor='green')
    plt.title(f"Histogram of {column_names[i]}")
    plt.xlabel(column_names[i])
    plt.ylabel("Frequency")
    plt.show()
    

    # Mean, Standard Deviation, Sample Size (n) of Sample 2
    
    x_bar2 = np.mean(sample2)
    stdv2 = np.std(sample2)
    n2 = np.size(sample2)
    print("\n Analysis of sample 1: Mean = %.2f, Std. Dev. = %.2f, n = %d" % (x_bar2, stdv2, n2))
    
    # Histogrtam of all the Variables of Sample 2 
    
    plt.hist(sample2, color='green', edgecolor='black')
    plt.title(f"Histogram of {column_names[i]}")
    plt.xlabel(column_names[i])
    plt.ylabel("Frequency")
    plt.show()

    # performing z-test for two sample size as the sample is independent, population is normally distributed, and sample size in large enough for central limit theoram
    # h0 (Null Hypethesis): mean of sample 1 = mean of sample 2
    # h1 (Alternative Hypothesis): mean of sample 1 is not equal to mean of sample 2
    # equal_var=False, indicates two populations do not have equal variance
    

    z_score = (x_bar1 - x_bar2) / np.sqrt((stdv1**2 / n1) + (stdv2**2 / n2))
    p_val = 2 * (1 - st.norm.cdf(np.abs(z_score)))

    print("\n Computing Z-score ...")
    print("\t Z-score: %.2f" % z_score)

    print("\n Computing p-value ...")
    print("\t p-value: %.4f" % p_val)

    print("\n Conclusion:")
    if p_val < 0.10:
      print("\t We reject the null hypothesis for", column_names[i])
    variable_analysis.append(column_names[i])
else:
    print("\t We accept the null hypothesis for", column_names[i])
    print()

print(variable_analysis)
print("Number of rejected null hypotheses:", len(variable_analysis))