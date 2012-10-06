Project on Machine Learning by Kedar Phadtare

Files/folders in this project:

spambase.data                     The Spambase data set consists of 4,601 e-mails, of which 1,813 are spam (39.4%). The data set archive contains a processed version of the e-mails wherein 57 real-valued features have been extracted and the 							spam/non-spam label has been assigned.
spam-filter.py			This file contains the implementation of three methods used for the classifications of emails in the entire spambase.data dataset, they being the Bernoulli Distribution, Gaussian Random Variable Distribution and 							the Histogram Distribution.
						To estimate the generalization (testing) error of the classifiers, cross-validation was performed. In k-fold cross-validation, one would ordinarily partition the data set randomly into k groups of roughly equal size and 							perform k experiments (the "folds") wherein a model is trained on k-1 of the groups and tested on the remaining group, where each group is used for testing exactly once. (A common value of k as 10 was chosen.) The 							generalization error of the classifier is estimated by the average of the performance across all k folds.
						While performing cross-validation with random partitions, for consistency and comparability of the results,  data was partitioned into 10 groups as follows: Consider the 4,601 data points in the order 							they appear in the processed data file. Each group will consist of every 10th data point in file order, i.e., Group 1 will consist of points {1,11,21,...}, Group 2 will consist of points {2,12,22,...}, ..., and Group 10 will consist 							of ponts {10,20,30,...}. Finally, Fold k will consist of testing on Group k a model obtained by training on the remaining k-1 groups.
						Creates the three Naive Bayes classifiers by modeling the features in three different ways.
						The 57 features are real-valued, and one can model the feature distributions in simple and complex ways. You will explore the effect of this modeling on overall performance.
						 - Modelled the features as simple Bernoulli (Boolean) random variables. Considered a threshold such as the overall mean value of the feature in the training set, and simply computed the fraction of 							the time that the feature value is above or below the overall mean value for each class. In other words, for feature fi with overall mean value mui, estimated
						-Pr[fi <= mui | spam]
						-Pr[fi > mui | spam]
						-Pr[fi <= mui | non-spam]
						-Pr[fi > mui | non-spam]
						and used these estimated values in the Naive Bayes predictor. Used Laplace smoothing to avoid issues with zero probabilities.
						- Modelled the features as Gaussian random variables, estimating the class conditional mean and variance as appropriate. Used the probability density function to calculate the conditional probabilities. Used laplace, and 							 jelinek mercer smoothing to take care of zero probabilities.
						- Modelled the feature distribution via a histogram. Bucketed the data into a number of bins and estimated the class conditional probabilities of each bin. For example, for any feature, consider the following values: 							min-value, mean-value, max-value, and the class conditional (spam/non-spam) mean values, one of which is presumably lower than the overall mean, and one of which is higher than the overall mean. Order these values 							numerically, and created four feature value bins as follows
						-[min-value, low-mean-value]
						-(low-mean-value, overall-mean-value]
						-(overall-mean-value, high-mean-value]
						-(high-mean-value, max-value]
						Based on the conditional probabilities of these bins, spam or not spam email were predicted.

P.S: Most of this text is obtained from the problem statement available on : http://www.ccs.neu.edu/home/jaa/CS6140.12F/Homeworks/hw.02.html
