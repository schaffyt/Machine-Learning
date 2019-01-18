"""
Adam Schaffroth
CPSC 392 Linstead Section 1
10/18/18
"""


import math

#valueSplit is a list of 2 items. The first is number of positive examples,
#the second is the number of negative examples
def calcEntropy(valueSplit):
    h = 0.0
    #fill this in
    p0 = valueSplit[0]/(valueSplit[0] + valueSplit[1])
    p1 = valueSplit[1]/(valueSplit[0] + valueSplit[1])
    h = -p0*math.log2(p0) - p1*math.log2(p1)
    return h

#should be .940
#rootValues is a list of the values at the parent node. It consists of 2 items.
#The first is number of positive examples,
#the second is the number of negative examples
#descendantValues is a list of lists.  Each inner list consists of the number of positive
#and negative examples for the attribute value you want to split on.
def calcInfoGain(rootValues,descendantValues):
    gain = 0.0
    #fill this in
    sum_weighted_entropy = 0.0
    for i in range(len(descendantValues)):
        sv_div_s = math.fabs(descendantValues[i][0] + descendantValues[i][1]) / math.fabs(rootValues[0] + rootValues[1])
        sum_weighted_entropy += sv_div_s* calcEntropy(descendantValues[i])
    gain = calcEntropy(rootValues) - sum_weighted_entropy
    return gain




if __name__ == "__main__":
    attributeName = "Humidity"
    rootSplit = [9,5] # 9 positive, 5 negative examples
    descendantSplit = [[3,4],[6,1]]
    ig = calcInfoGain(rootSplit, descendantSplit)
    print("The information gain of splitting on ",attributeName," is: ",ig," bits")
    
    
