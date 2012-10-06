#!/usr/bin/env python
from decimal import *
from math import *

#-----------------------------------------------------------------------------------Helper functions common for all methods - begin-----------------------------------------------------------------------------------------------------------------------------

#Calculates the conditional mean values for the features in the training set spam/not spam emails.
#input is the training set and returns the mean values dictionary and a list of emails.
#Helper function called by all the 3 classifiers.
#contract -> (list) returns (dict, dict, int,int)
def calculateMeanValues(trainingGroup):
	trainEmail = []
	meanValuesForSpam = {}
	meanValuesForNotSpam = {}	
	for groups in trainingGroup:
		for email in groups:
			trainEmail.append(email)	
	for feature in range(0,57):
		spamCount = 0
		notSpamCount = 0
		featureMeanForSpam = 0
		featureMeanForNotSpam = 0
		for email in trainEmail:
			email = email.split(',')
			if int(email[57]) == 1:	
				spamCount += 1
				featureMeanForSpam += float(email[feature])
			else:
				notSpamCount += 1
				featureMeanForNotSpam += float(email[feature])
		meanValuesForSpam[feature] = featureMeanForSpam/spamCount
		meanValuesForNotSpam[feature] = featureMeanForNotSpam/notSpamCount
	return meanValuesForSpam,meanValuesForNotSpam,trainEmail,spamCount,notSpamCount

#This function calculates the error rate of the predictions obtained from calcuateBernoulli function.
#Takes a testing group and a calculated dict (from main method function), prints and returns the error rate, FPR and FNR.
#Helper function called by all the 3 classifiers.
#contract -> (list,dict) returns (float,dict,float,float)
def calculateErrorRate(testingGroup,resultDict):		
	actualSpamCount = 0
	actualNotSpamCount = 0
	actualDict = {}
	loopCounter = 0
	for email in testingGroup:
		email = email.split(",")
		if int(email[57]) == 1:
			actualDict[loopCounter] = 1			
			actualSpamCount += 1
		else:
			actualDict[loopCounter] = 0
			actualNotSpamCount += 1
		loopCounter += 1
	truePositiveRate = 0.0
	falsePositiveRate = 0.0
	trueNegativeRate = 0.0
	falseNegativeRate = 0.0
	for i in range(0,len(testingGroup)):
		if (resultDict[i] == 1) and (actualDict[i]== 1):
			truePositiveRate += 1
		elif (resultDict[i] == 1) and (actualDict[i] == 0):
			falsePositiveRate += 1
		elif (resultDict[i] == 0) and (actualDict[i] == 1):
			falseNegativeRate += 1
		elif (resultDict[i] == 0) and (actualDict[i] == 0):
			trueNegativeRate += 1
	errorRate = float(falseNegativeRate+falsePositiveRate)/float(falseNegativeRate+falsePositiveRate+truePositiveRate+trueNegativeRate)
	print "Error Rate = %s , FPR = %s , FNR = %s  "  %(errorRate ,falsePositiveRate/(falsePositiveRate+trueNegativeRate),falseNegativeRate/(falseNegativeRate+truePositiveRate))
	return errorRate,actualDict,falsePositiveRate/(falsePositiveRate+trueNegativeRate),falseNegativeRate/(falseNegativeRate+truePositiveRate) #print this to a file for graphing purposes.

#Function calculates the area under the ROC based on the points in the tpr and fpr list.
#the fpr is taken as x axis and the tpr is taken on the y axis.
#prints a float area.
#contract -> (list,list) returns (void), prints the area.
def getArea(tprList,fprList):
	area = 0.0
	for element in range(1,len(tprList)):
		area += (fprList[element]-fprList[element-1])*(tprList[element]+tprList[element-1])
	area = 0.5*area
	print "Area under the ROC = " ,area
	print "----------------------------------------------------------------------------"

#gets predictions for the new threshold value and compares with actual values.
#caclulates the tp,fp,fn,tn,tpr,fpr values and makes the tprList and fprList needed to calculate the area.
#gets points for the ROC
#contract -> (list,dict,list) returns (void)
def getROCPoints(scoreList,actualDict,testingGroup):
	thresholdList = sorted(scoreList)
	thresholdList = reversed(thresholdList)
	for threshold in thresholdList:
		resultDict = {}		
		for i in range(0,len(testingGroup)):
			resultDict[i] = 0
		loopCounter = 0
		for score in scoreList:	
			truePositiveRate = 0.0
			falsePositiveRate = 0.0
			falseNegativeRate = 0.0
			trueNegativeRate = 0.0
			if score == threshold: #dont compare the threshold with itself.
				pass 
			if score > threshold:
				resultDict[loopCounter] = 1 # if score is greater than threshold, mark the email as spam.
			loopCounter += 1	
		for i in range(0,len(testingGroup)): # comparing the predicted and actual values.
			if (resultDict[i] == 1) and (actualDict[i]== 1):
				truePositiveRate += 1
			elif (resultDict[i] == 1) and (actualDict[i] == 0):
				falsePositiveRate += 1
			elif (resultDict[i] == 0) and (actualDict[i] == 1):
				falseNegativeRate += 1
			elif (resultDict[i] == 0) and (actualDict[i] == 0):
				trueNegativeRate += 1
		tprList.append(truePositiveRate/(truePositiveRate+falseNegativeRate))
		fprList.append(falsePositiveRate/(falsePositiveRate+trueNegativeRate))
	getArea(tprList,fprList)

#-----------------------------------------------------------------------------------Helper functions common for all methods - end-----------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------Helper functions for Bernoulli method - begin-----------------------------------------------------------------------------------------------------------------------------

#Works with the training set and the mean values, classifying the email probabilities according to spam/not spam and feature/meanAverageValue comparison.
#Pr[fi <= mui | spam]
#Pr[fi > mui | spam]
#Pr[fi <= mui | non-spam]
#Pr[fi > mui | non-spam]
#Takes training email list and mean values dictionary and returns 4 dictionaries of probabilities which are caclulated using the above formulae. It also returns the spam and ham count for the training set.
#called by the naive bayes classisifer using bernoulli distribution.
#contract -> (list,dict,int,int) returns (dict,dict,dict,dict)
def normalizeTrainingSet(trainEmail,meanValues,spamCount,notSpamCount):
	probabilitiesCaseI = {}
	probabilitiesCaseII = {}
	probabilitiesCaseIII = {}
	probabilitiesCaseIV = {}
	for i in range(0,57):
		probabilitiesCaseI[i] = 0.0
		probabilitiesCaseII[i] = 0.0
		probabilitiesCaseIII[i] = 0.0
		probabilitiesCaseIV[i] = 0.0
	for email in trainEmail:
		email = email.split(',')
		for feature in range(0,57):
			if (float(email[feature])<=meanValues[feature]) and int(email[57])==1:
				probabilitiesCaseI[feature] += 1
			elif (float(email[feature])>meanValues[feature]) and int(email[57])==1:
				probabilitiesCaseII[feature] += 1
			elif (float(email[feature])<=meanValues[feature]) and int(email[57])==0:
				probabilitiesCaseIII[feature] += 1
			elif (float(email[feature])>meanValues[feature]) and int(email[57])==0:
				probabilitiesCaseIV[feature] += 1
	for i in range(0,57):
		probabilitiesCaseI[i] = (probabilitiesCaseI[i]+1)/(float(spamCount) + 2)
		probabilitiesCaseII[i] = (probabilitiesCaseII[i]+1)/(float(spamCount) + 2)
		probabilitiesCaseIII[i] = (probabilitiesCaseIII[i]+1)/(float(notSpamCount) + 2)
		probabilitiesCaseIV[i] = (probabilitiesCaseIV[i]+1)/(float(notSpamCount) + 2)
	return probabilitiesCaseI,probabilitiesCaseII,probabilitiesCaseIII,probabilitiesCaseIV

#This function takes the testing/training groups,set of probabilites (calcuated in normalizeTrainingSet function), spamCount/notSpamCount (for the training set) and predicts spam/notSpam for the testing set.
#Returns the total spam/notSpam count, and a dictionary with the predicted values based on scores calcuated from the formula.
#called by the naive bayes classisifer using bernoulli distribution.
#contract -> (list, dict, dict, dict, dict, dict, int, int) returns (dict, list)
def bernoulliCalculation(testingGroup,meanValues,probabilitiesCaseI,probabilitiesCaseII,probabilitiesCaseIII,probabilitiesCaseIV,spamCount,notSpamCount):
	bernoulliDict = {}
	scoreList = []
	for i in range(0,len(testingGroup)):
		bernoulliDict[i]=1	
	constantProbability = log(float(spamCount)/float(notSpamCount))
	loopCounter = 0
	for email in testingGroup:
		variantProbability = 0
		email = email.split(',')		
		for feature in range(0,57):
			if ((float(email[feature]) <= meanValues[feature]) and (float(email[feature]) <= meanValues[feature])):
				variantProbability += log(float(probabilitiesCaseI[feature])/float(probabilitiesCaseIII[feature]))
			elif ((float(email[feature]) <= meanValues[feature]) and (float(email[feature]) > meanValues[feature])):
				variantProbability += log(float(probabilitiesCaseI[feature])/float(probabilitiesCaseIV[feature]))
			elif ((float(email[feature]) > meanValues[feature]) and (float(email[feature]) <= meanValues[feature])):
				variantProbability += log(float(probabilitiesCaseII[feature])/float(probabilitiesCaseIII[feature]))
			else:
				variantProbability += log(float(probabilitiesCaseII[feature])/float(probabilitiesCaseIV[feature]))
		score = constantProbability + variantProbability
#		processScore(score)	
		if score < 0:
			bernoulliDict[loopCounter] = 0
		loopCounter += 1	
		scoreList.append(score)
	return bernoulliDict,scoreList

#-----------------------------------------------------------------------------------Helper functions for Bernoulli method - end-----------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------Helper functions for Gaussian method - begin-----------------------------------------------------------------------------------------------------------------------------

#calculates the mean values for the features in the training set.
#takes as input the entire training set of emails.
#returns a idctionary of mean values for all features of the training set.
#contract -> (list) returns (dict)
def calculateGaussianMean(gaussianTrainEmail):
	gaussianMeanValues = {}
	for i in range(0,57):
		gaussianMeanValues[i] = 0
	for email in gaussianTrainEmail:
		email = email.split(",")
		for feature in range(0,57):
			gaussianMeanValues[feature] += float(email[feature])
	for i in range(0,57):
		gaussianMeanValues[i] /= len(gaussianTrainEmail)
	return gaussianMeanValues 
	
#calculates the variance for gaussian distribution based on conditional probabilities for spam and not spam. Returns variance for the data given spam, and variance given not spam.
#input is the conditional mean values for spam/not spam, overall mean values, training set.
#output is the conditional variance for spam/not spam for all the features in the emails of the training set.
#contract -> (dict, dict, dict, list) returns (dict, dict)
def calculateGaussianVariance(gaussianMeanValuesForSpam,gaussianMeanValuesForNotSpam,gaussianMeanValues,gaussianTrainEmail):
	gaussianVarianceForSpam = {}
	gaussianVarianceForNotSpam = {}
	overallVariance = {}
	spamCount = 0
	notSpamCount = 0
	for i in range(0,57):
		gaussianVarianceForSpam[i] = 0.0
		gaussianVarianceForNotSpam[i] = 0.0
		overallVariance[i] = 0.0
	for email in gaussianTrainEmail:
		email = email.split(",")
		if int(email[57]) == 1:
			spamCount += 1
			for feature in range(0,57):
				overallVariance[feature] +=  (float(email[feature]) - float(gaussianMeanValues[feature]))**2
				gaussianVarianceForSpam[feature] += (float(email[feature]) - float(gaussianMeanValuesForSpam[feature]))**2
		else:
			notSpamCount += 1
			for feature in range(0,57):
				overallVariance[feature] +=  (float(email[feature]) - float(gaussianMeanValues[feature]))**2
				gaussianVarianceForNotSpam[feature] += (float(email[feature]) - float(gaussianMeanValuesForNotSpam[feature]))**2
	for feature in range(0,57):
		gaussianVarianceForSpam[feature] = (float(gaussianVarianceForSpam[feature]))/(float(spamCount))
		gaussianVarianceForNotSpam[feature] = (float(gaussianVarianceForNotSpam[feature]))/(float(notSpamCount))
		overallVariance[feature] /= float(spamCount+notSpamCount)
		if (gaussianVarianceForSpam[feature] == 0 and gaussianVarianceForNotSpam[feature] == 0 and overallVariance[feature] == 0):
			gaussianVarianceForSpam[feature] = 1 
        		gaussianVarianceForNotSpam[feature] = 1
			overallVariance[feature] = 1
		else:
			gaussianVarianceForSpam[feature] = 0.2*gaussianVarianceForSpam[feature] + 0.8*overallVariance[feature]
			gaussianVarianceForNotSpam[feature] = 0.2*gaussianVarianceForNotSpam[feature] + 0.8*overallVariance[feature]
	return gaussianVarianceForSpam,gaussianVarianceForNotSpam

#predicts whether the emails in the testing group is spam or not and returns a dictionary with values 1 for spam and 0 for not spam for each email.	
#input is the testing group, conditional variance dictionaries for spam/not spam, conditional mean values for spam/not spam, and spam/not spam count for that fold.
#returns the score list with the calculated scores on basis of the gaussian PDF and the predicted testing test results dictionary.
#contract -> (list,dict,dict,dict,dict,int,int) returns (dict,list)
def calculateGaussianProbability(testingGroup,gaussianVarianceForSpam,gaussianVarianceForNotSpam,gaussianMeanValuesForSpam,gaussianMeanValuesForNotSpam,gaussianSpamCount,gaussianNotSpamCount):
	constantProbability = log(float(gaussianSpamCount)/float(gaussianNotSpamCount))
	gaussianPredictionDict = {}
	scoreList = []
	for i in range(0,len(testingGroup)):
		gaussianPredictionDict[i] = 1
	loopCounter = 0
	for email in testingGroup:
		variantComponent = 0.0
		email = email.split(",")
		for feature in range(0,57): #calculating the probability of feature being spam and not spam given data with the simplified gaussian formula.
			variantComponentSpam = (-0.5*log(2*pi) - ((float(email[feature]) - float(gaussianMeanValuesForSpam[feature]))**2)/(2*(float(gaussianVarianceForSpam[feature])**2)) - log(float(gaussianVarianceForSpam[feature])))
			variantComponentNotSpam = (-0.5*log(2*pi) - ((float(email[feature]) - float(gaussianMeanValuesForNotSpam[feature]))**2)/(2*(float(gaussianVarianceForNotSpam[feature])**2)) - log(float(gaussianVarianceForNotSpam[feature])))
			variantComponent += (float(variantComponentSpam) - float(variantComponentNotSpam))
		score = constantProbability + variantComponent
		if score < 0: #if the score is less than 0, mark the email as ham.
			gaussianPredictionDict[loopCounter] = 0
		loopCounter += 1
		scoreList.append(score)
	return gaussianPredictionDict,scoreList

#-----------------------------------------------------------------------------------Helper functions for Gaussian method - end-----------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------Helper functions for histogram method - begin--------------------------------------------------------------------------------------------------------------------------

#calculates the minimum and maximum values in the training set for each feature.
#input is the entire training set.
#returns the minimum values dictionary and the maximum values dictionary.
#contract -> (list) returns (dict,dict)
def calculateMinMaxValues(histogramTrainEmail):
	histogramMinValues = {}
	histogramMaxValues = {}
	for i in range(0,57):
		histogramMinValues[i] = 0.0
		histogramMaxValues[i] = 0.0
	for feature in range(0,57):
		valuesList = []
		for email in histogramTrainEmail:
			email = email.split(",")
			valuesList.append(float(email[feature]))
		histogramMinValues[feature] = min(valuesList)
		histogramMaxValues[feature] = max(valuesList)
		del valuesList[:]
	return histogramMinValues,histogramMaxValues

#returns the low-mean-value.
#input is the mean value for spam, overall mean value, mean value for not spam.
#returns float mean value for either spam/not spam.
#contract -> (float,float,float) returns (float)
def calcLowMean(histogramMeanValuesForSpam,histogramMeanValues,histogramMeanValuesForNotSpam):
	if (histogramMeanValuesForSpam < histogramMeanValues):
		return histogramMeanValuesForSpam
	else:
		return histogramMeanValuesForNotSpam

#returns the high-mean-value.
#input is the mean value for spam, overall mean value, mean value for not spam.
#returns float mean value for either spam/not spam.
#contract -> (float,float,float) returns (float)
def calcHighMean(histogramMeanValuesForSpam,histogramMeanValues,histogramMeanValuesForNotSpam):
	if (histogramMeanValuesForSpam > histogramMeanValues):
		return histogramMeanValuesForSpam
	else:
		return histogramMeanValuesForNotSpam

#creates the 4 bins based on the 4 conditions to be checked.
#bin1 =[min-value, low-mean-value]
#bin2 =(low-mean-value, overall-mean-value]
#bin3 =(overall-mean-value, high-mean-value]
# bin4 = (high-mean-value, max-value]
#input is the conditional mean values for spam/not spam, min/max values, overall mean values, training set, spam/not spam count.
#output is the conditional dictionaries for spam and not spam containing the 4 bin probabilities.
#contract -> (dict,dict,dict,dict,dict,list,int,int) returns (dict[list(float)], dict[list(float)])
def createBins(histogramMeanValuesForSpam,histogramMeanValuesForNotSpam,histogramMinValues,histogramMaxValues,histogramMeanValues,histogramTrainEmail,histogramSpamCount,histogramNotSpamCount):
	spamBin = {}
	notSpamBin = {}
	for i in range(0,57):
		spamBin[i] = 0
		notSpamBin[i] = 0
	for feature in range(0,57):
		bin1Spam = 0.0
		bin2Spam = 0.0
		bin3Spam = 0.0
		bin4Spam = 0.0
		bin1NotSpam = 0.0
		bin2NotSpam = 0.0
		bin3NotSpam = 0.0
		bin4NotSpam = 0.0
		for email in histogramTrainEmail:
			email = email.split(",")
			if int(email[57]) ==1: #calculate the conditional probabilities for the features of the spam emails.
				if ((float(email[feature]) >= histogramMinValues[feature]) and (float(email[feature]) <= calcLowMean(histogramMeanValuesForSpam[feature],histogramMeanValues[feature],histogramMeanValuesForNotSpam[feature]))):
					bin1Spam += 1
				elif ((float(email[feature]) > calcLowMean(histogramMeanValuesForSpam[feature],histogramMeanValues[feature],histogramMeanValuesForNotSpam[feature])) and (float(email[feature]) <= histogramMeanValues[feature])):
					bin2Spam += 1
				elif ((float(email[feature]) > histogramMeanValues[feature]) and (float(email[feature]) <= calcHighMean(histogramMeanValuesForSpam[feature],histogramMeanValues[feature],histogramMeanValuesForNotSpam[feature]))):
					bin3Spam += 1
				elif ((float(email[feature]) > calcHighMean(histogramMeanValuesForSpam[feature],histogramMeanValues[feature],histogramMeanValuesForNotSpam[feature])) and (float(email[feature]) <= histogramMaxValues[feature])):
					bin4Spam += 1
			else: #calculate the conditional probabilities for the features of the not spam emails.
				if ((float(email[feature]) >= histogramMinValues[feature]) and (float(email[feature]) <= calcLowMean(histogramMeanValuesForSpam[feature],histogramMeanValues[feature],histogramMeanValuesForNotSpam[feature]))):
					bin1NotSpam += 1
				elif ((float(email[feature]) > calcLowMean(histogramMeanValuesForSpam[feature],histogramMeanValues[feature],histogramMeanValuesForNotSpam[feature])) and (float(email[feature]) <= histogramMeanValues[feature])):
					bin2NotSpam += 1
				elif ((float(email[feature]) > histogramMeanValues[feature]) and (float(email[feature]) <= calcHighMean(histogramMeanValuesForSpam[feature],histogramMeanValues[feature],histogramMeanValuesForNotSpam[feature]))):
					bin3NotSpam += 1
				elif ((float(email[feature]) > calcHighMean(histogramMeanValuesForSpam[feature],histogramMeanValues[feature],histogramMeanValuesForNotSpam[feature])) and (float(email[feature]) <= histogramMaxValues[feature])):
					bin4NotSpam += 1
		spamBin[feature] = [(bin1Spam+1)/(float(histogramSpamCount)+2), (bin2Spam+1)/(float(histogramSpamCount)+2), (bin3Spam+1)/(float(histogramSpamCount)+2),(bin4Spam+1)/(float(histogramSpamCount)+1)]
		notSpamBin[feature] = [(bin1NotSpam+1)/(float(histogramNotSpamCount)+2), (bin2NotSpam+1)/(float(histogramNotSpamCount)+2), (bin3NotSpam+1)/(float(histogramNotSpamCount)+2), (bin4NotSpam+1)/(float(histogramNotSpamCount)+2)]
	return spamBin,notSpamBin

#predicts whether the emails in the testing group is spam or not and returns a dictionary with values 1 for spam and 0 for not spam for each email.
#input is the spam/not spam bin dicts, testing group, spam/not spam counts, mean values for spam/not spam dicts, min/max values dict, overall mean values dict.
#returns the prediction dict and score list.
#contract -> (dict[list(float)],dict[list(float)],list,int,int,dict,dict,dict,dict,dict,dict,dict) returns (dict,list)
def calculateHistogramDict(spamBin,notSpamBin,testingGroup,histogramSpamCount,histogramNotSpamCount,histogramMeanValuesForSpam,histogramMeanValuesForNotSpam,histogramMinValues,histogramMaxValues,histogramMeanValues):
	histogramDict = {}
	scoreList = []
	constantProbability = log(float(histogramSpamCount)/float(histogramNotSpamCount))
	for i in range(0,len(testingGroup)):
		histogramDict[i] = 1
	loopCounter = 0
 	for email in testingGroup:
		variantProbability = 0.0
		email = email.split(",")
		for feature in range(0,57): #calculate the probabilities for the features of the email.
			if ((float(email[feature]) >= histogramMinValues[feature]) and (float(email[feature]) <= calcLowMean(histogramMeanValuesForSpam[feature],histogramMeanValues[feature],histogramMeanValuesForNotSpam[feature]))):
				variantProbability += log((spamBin[feature][0])/(notSpamBin[feature][0]))
			elif ((float(email[feature]) > calcLowMean(histogramMeanValuesForSpam[feature],histogramMeanValues[feature],histogramMeanValuesForNotSpam[feature])) and (float(email[feature]) <= histogramMeanValues[feature])):
				variantProbability += log((spamBin[feature][1])/(notSpamBin[feature][1]))
			elif ((float(email[feature]) > histogramMeanValues[feature]) and (float(email[feature]) <= calcHighMean(histogramMeanValuesForSpam[feature],histogramMeanValues[feature],histogramMeanValuesForNotSpam[feature]))):
				variantProbability += log((spamBin[feature][2])/(notSpamBin[feature][2]))
			elif ((float(email[feature]) > calcHighMean(histogramMeanValuesForSpam[feature],histogramMeanValues[feature],histogramMeanValuesForNotSpam[feature])) and (float(email[feature]) <= histogramMaxValues[feature])):
				variantProbability += log((spamBin[feature][3])/(notSpamBin[feature][3]))
		score = constantProbability + variantProbability
#		print score
		if score < 0: # if score is less than 0, then mark it as ham.
			histogramDict[loopCounter] = 0		
		loopCounter += 1
		scoreList.append(score)
	return histogramDict,scoreList

#-----------------------------------------------------------------------------------Helper functions for histogram method - end-------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------main functions for the three methods - begin---------------------------------------------------------------------------------------------------------------------

#This function takes in the training and testing sets and performs the training/testing operations for Histogram classifier and calcuates the error rate,tpr,fpr by calling helper functions.
#input is the testing group and the training group of emails.
#prints the error rate,false positive rate, false negative rate, and the area under the curve for a particular fold.
#contract -> (list,list) returns (void) , like a main function for the histogram method. calls other functions and gets results printed.
def histogramMethod(testingGroup,trainingGroup):
	global tprList 
	tprList = []
	global fprList 	
	fprList = []	
	histogramMeanValuesForSpam,histogramMeanValuesForNotSpam,histogramTrainEmail,histogramSpamCount,histogramNotSpamCount = calculateMeanValues(trainingGroup)
	histogramMinValues,histogramMaxValues = calculateMinMaxValues(histogramTrainEmail)
	histogramMeanValues = calculateGaussianMean(histogramTrainEmail)
	spamBin,notSpamBin = createBins(histogramMeanValuesForSpam,histogramMeanValuesForNotSpam,histogramMinValues,histogramMaxValues,histogramMeanValues,histogramTrainEmail,histogramSpamCount,histogramNotSpamCount)
	histogramDict,scoreList =calculateHistogramDict(spamBin,notSpamBin,testingGroup,histogramSpamCount,histogramNotSpamCount,histogramMeanValuesForSpam,histogramMeanValuesForNotSpam,histogramMinValues,histogramMaxValues,histogramMeanValues)
	errorRate,actualDict,falsePositiveRate,falseNegativeRate = calculateErrorRate(testingGroup,histogramDict)
	getROCPoints(scoreList,actualDict,testingGroup)

#This function takes in the training and testing sets and performs the training/testing operations forGaussian classifier and calcuates the error rate,tpr,fpr by calling helper functions.
#input is the testing group and the training group of emails.
#prints the error rate,false positive rate, false negative rate, and the area under the curve for a particular fold.
#contract -> (list,list) returns (void) , like a main function for the gaussian method. calls other functions and gets results printed.
def gaussianMethod(testingGroup,trainingGroup):
	global tprList 
	tprList = []
	global fprList 	
	fprList = []	
	gaussianMeanValuesForSpam,gaussianMeanValuesForNotSpam,gaussianTrainEmail,gaussianSpamCount,gaussianNotSpamCount = calculateMeanValues(trainingGroup)
	gaussianMeanValues = calculateGaussianMean(gaussianTrainEmail)
	gaussianVarianceForSpam,gaussianVarianceForNotSpam = calculateGaussianVariance(gaussianMeanValuesForSpam,gaussianMeanValuesForNotSpam,gaussianMeanValues,gaussianTrainEmail)
	gaussianPredictionDict,scoreList = calculateGaussianProbability(testingGroup,gaussianVarianceForSpam,gaussianVarianceForNotSpam,gaussianMeanValuesForSpam,gaussianMeanValuesForNotSpam,gaussianSpamCount,gaussianNotSpamCount)
	errorRate,actualDict,falsePositiveRate,falseNegativeRate = calculateErrorRate(testingGroup,gaussianPredictionDict)
	getROCPoints(scoreList,actualDict,testingGroup)

#This function takes in the training and testing sets and performs the training/testing operations for bernoulli classifier and calcuates the error rate,tpr,fpr by calling helper functions.
#input is the testing group and the training group of emails.
#prints the error rate,false positive rate, false negative rate, and the area under the curve for a particular fold.
#contract -> (list,list) returns (void), like the main function for the bernoulli method. Works by calling helper functions.
def training(testingGroup,trainingGroup):
	global tprList 
	tprList = []
	global fprList 	
	fprList = []	
	meanValuesForSpam,meanValuesForNotSpam,trainEmail,spamCount,notSpamCount = calculateMeanValues(trainingGroup)
	meanValues = calculateGaussianMean(trainEmail)
	probabilitiesCaseI,probabilitiesCaseII,probabilitiesCaseIII,probabilitiesCaseIV = normalizeTrainingSet(trainEmail,meanValues,spamCount,notSpamCount)	
	resultDict,scoreList = bernoulliCalculation(testingGroup,meanValues,probabilitiesCaseI,probabilitiesCaseII,probabilitiesCaseIII,probabilitiesCaseIV,spamCount,notSpamCount)
	errorRate,actualDict,falsePositiveRate,falseNegativeRate = calculateErrorRate(testingGroup,resultDict)
	getROCPoints(scoreList,actualDict,testingGroup)

#-----------------------------------------------------------------------------------main functions for the three methods - end---------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------Main function for the program and K Fold creation - begin--------------------------------------------------------------------------------------------------

#reads the spambase file and creates the k folds.
#input is nothing, output are the groups.
#contract -> (void) returns (list[list])
def getKFolds():
#gets all the data from the spambase file and stores it in a dictionary
	spambaseData = open('spambase.data','r').readlines()
	spambase = {}
	i = 1
	for spamfile in spambaseData:
		spambase[i] = spamfile.strip()
		i += 1
#breaks the spambase dictonary into smaller chunks of sort {1,11,22,...} uptil {10,20,30,...}
	groups = []
	for k in range(1,11):
		fold = []
		fold.append(spambase[k])
		j=k
		while(j<4599):
			if(j+10>4601):
				break
			else:
				fold.append(spambase[j+10])
				j += 10
		groups.append(fold)
	return groups

#This function takes a number, and separates it from the total groups, creating a testing group and a training group.
#Takes a set number to be excluded (iteration wise) and passed 2 lists : testing group and training group to the training function.
#Based on the input option, calls the appropriate function.
#contract -> (int,list[list],int) returns (void) ,like a main function for the entire 3 methods. calls the related functions with corresponding parameters.
def trainFolds(excludeSet,groups,option):
	trainingGroup = []
	testingGroup = groups[excludeSet]
	for j in range(0,len(groups)):
		if j != excludeSet:
			trainingGroup.append(groups[j])
	if option == 1:
		print "Results for Fold " , excludeSet
		training(testingGroup,trainingGroup)                    #Runs Bernoulli Method.
	elif option == 2:
		print "Results for Fold " , excludeSet
		gaussianMethod(testingGroup,trainingGroup)	#Runs Gaussian Method.
	elif option ==3:
		print "Results for Fold " , excludeSet
		histogramMethod(testingGroup,trainingGroup) #Runs Histogram Method.

#Kicks off the process of training and testing for the 10 folds by first creating the folds.
#validates the input and sends parameters to functions.
def main():
	groups = getKFolds()
 	option = int(input("Enter: \n 1 to run the Bernoulli method \n 2 for the Gaussian method and  \n 3 for Histogram Method."))
	if option != 1 and option!=2 and option!=3:
	 	print("Invalid Input, please run again with proper inputs.")
	if option == 1:
		print "Results for Naive Bayes Filter using Bernoulli Distribution."
	if option == 2:
		print "Results for Naive Bayes Filter using Gaussian Distribution."
	if option == 3:
		print "Results for Naive Bayes Filter using Histogram Distribution."
	for excludeSet in range(0,len(groups)):
#	for excludeSet in range(0,1):   #runs only one set.
		trainFolds(excludeSet,groups,option)	

#calling the main.
if __name__ == "__main__":
	main()
#-----------------------------------------------------------------------------------Main function for the program and K Fold creation - end--------------------------------------------------------------------------------------------------
