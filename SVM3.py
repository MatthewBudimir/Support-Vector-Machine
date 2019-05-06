
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
import cvxopt
import cvxopt.solvers
import sys


def linear(x1, x2):
    return np.dot(x1, x2)

def preprocess_data(data):
    for index in range(len(data)):
        if data[index][0] == 0:
            data[index][0] = -1
    return data

class SupportVectorMachine(object):
    def __init__(self):
        self.kernel = linear;

    def trainPrimalHard(self,data):
        size = len(data);
        # Gather your X's
        x = np.array([row[1:] for row in data[0:]])

        # Append 1 to all X's for b
        x = np.append(x, np.ones(size)[:,None], axis=1)
        width = len(x[0])
        # print(len(x))
        # Gather your Y's
        y = np.array([row[0] for row in data])
        # X transpose X is what we want, so we don't need anything for P so P is the identity matrix that removes b from the input.
        P = np.identity(width);
        P[width-1][width-1] = 0;
        # We don't have a Q term for primal
        q = np.zeros(width);
        # A and B are zeros because we don't have any equals constraints
        A = np.zeros((size,width))
        b = np.zeros((size))
        # Calculate G:
        # Contains Multipliers for W0 through Wn and b (+1 is for b)
        G = np.zeros((size,width))
        for i in range(size):
            for j in range(width): #Don't do the last column since it applies to B.
                G[i][j] = y[i]*(x[i][j])

        # Constraint is that no points are in the gutter: therefore, h = [1,1,1,1,1....]
        h = np.ones(size)
        # Constraints have '>=' instead of '<=' so multiply both sides by -1
        G = G*-1.0
        h = h*-1
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        A = cvxopt.matrix(A)
        G = cvxopt.matrix(G,(size,width),'d')
        h = cvxopt.matrix(h)
        b = cvxopt.matrix(b)
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b,kktsolver='ldl', options={'kktreg':1e-9});
        self.w = solution['x'][:-1];
        self.b = solution['x'][-1];

    def trainPrimalSoft(self,data,C):
        size = len(data);
        # Gather your X's
        x = np.array([row[1:] for row in data[0:]])
        # Append 1 to all X's for b
        x = np.append(x, np.ones(size)[:,None], axis=1)
        # Append Size 1's to all X's for soft margin deltas
        x = np.append(x, np.ones((size,size)), axis=1)
        width = len(x[0])
        wWidth = width-size;
        # Gather your Y's
        y = np.array([row[0] for row in data])
        # X transpose X is what we want, so we don't need anything for P so P is the identity matrix that removes b and delta from the input.
        P = np.identity(width);
        for i in range(wWidth-1,width):
            P[i][i] = 0;
        # Q term set up to calculate sum of all deltas multiplied by C.
        q = np.concatenate((np.zeros(wWidth),np.ones(size)*C));
        # A and B are zeros because we don't have any equals constraints
        A = np.zeros((size*2,width))
        b = np.zeros((size*2))
        # Calculate G:
        # Contains Multipliers for W0 through Wn and b (+1 is for b)
        G = np.zeros((size*2,wWidth))
        # G[i][j] = Yi*X[i][j]
        for i in range(size):
            for j in range(wWidth): #Don't do the last column since it applies to B.
                G[i][j] = y[i]*(x[i][j])
        # Append ones matrix with identity matrix on side.
        temp = np.concatenate((np.ones((size,size)),np.identity(size)),axis=0)
        # print(temp.shape)
        G = np.concatenate((G,temp),axis=1)
        # Constraint is that no points are in the gutter: therefore, h = [1,1,1,1,1....]
        h = np.ones(size)
        h = np.concatenate((h,np.zeros(size)));
        # Constraints have '>=' instead of '<=' so multiply both sides by -1
        G = G*-1.0
        h = h*-1
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        A = cvxopt.matrix(A)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        b = cvxopt.matrix(b)
        # run CVXOPT.
        solution = cvxopt.solvers.qp(P, q, G, h, A, b,kktsolver='ldl', options={'kktreg':1e-9});
        # read off results
        self.w = solution['x'][:200];
        self.b = solution['x'][201];

    def trainDualHard(self,data):
        size = len(data);
        # Gather Xs
        x = np.array([row[1:] for row in data[0:]])
        width = len(x[0])
        # Gather your Y's
        y = np.array([row[0] for row in data])
        # Multiplied by negative one to convert maximise problem into minimize problem.
        q = np.ones(size)*-1;

        # sum of all yixAi = 0 is just y dot alpha = 0.
        A = y;
        b = [0.0]
        # Calculate G:
        # Contains Multipliers for W0 through Wn and b (+1 is for b)
        # We want to include the dot product of xi,xj and yiyj in our ai.aj calculation
        print("calculating Kernels")
        sys.stdout.write("00.000%\r");
        kernelsCalculated = 0;
        P = np.zeros((size,size));
        for i in range(size):
            for j in range(size):
                P[i][j] = y[i]*y[j]*(self.kernel(x[i],x[j]))
                kernelsCalculated = kernelsCalculated+1;
                sys.stdout.write(str(round(float(100*kernelsCalculated)/(size*size),2)) + "%       \r")

        # GX < h is equivalent to every -alpha < 0. G is a negative identity matrix, h is just zeros
        G = np.identity(size)*-1;
        h = np.zeros(size);
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        A = cvxopt.matrix(A, (1,size)) #CVXOPT is finicky about this for some reason.
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        b = cvxopt.matrix(b)

        # Run optimization solver
        solution = cvxopt.solvers.qp(P, q, G, h, A, b,kktsolver='ldl', options={'kktreg':1e-9});
        self.w = np.zeros(width)
        self.alpha = solution['x'];
        for i in range(size):
            self.w += x[i]*self.alpha[i]*y[i]
        print(self.w.shape)
        for i in range(size):
            if self.alpha[i]>0:
                self.b =1/y[i] - np.dot(self.w,x[i]);
                break;

    def trainDualSoft(self,data,C):
        size = len(data);
        # Gather Xs
        x = np.array([row[1:] for row in data[0:]])
        width = len(x[0])
        # Gather your Y's
        y = np.array([row[0] for row in data])
        # Multiplied by negative one to convert maximise problem into minimize problem.
        q = np.ones(size)*-1;

        # sum of all yixAi = 0 is just y dot alpha = 0.
        A = y;
        b = [0.0]
        # Calculate G:
        # Contains Multipliers for W0 through Wn and b (+1 is for b)
        # G[i][j] = Yi*X[i][j]
        # We want to include the dot product of x and yiyj in our ai.aj
        print("calculating Kernels")
        sys.stdout.write("00.000%\r");
        kernelsCalculated = 0;
        P = np.zeros((size,size));
        for i in range(size):
            for j in range(size): #Don't do the last column since it applies to B.
                P[i][j] = y[i]*y[j]*(self.kernel(x[i],x[j]))
                kernelsCalculated = kernelsCalculated+1;
                sys.stdout.write(str(round(float(100*kernelsCalculated)/(size*size),2)) + "%       \r")

        # GX is a negative I matrix (for alpha > 0) stacked with an I matrix
        G = np.concatenate((np.identity(size)*-1,np.identity(size)), axis=0);
        h = np.concatenate((np.zeros(size),np.ones(size)*C));
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        A = cvxopt.matrix(A, (1,size)) #CVXOPT is finicky about this for some reason.
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        b = cvxopt.matrix(b)

        # Run optimization solver
        solution = cvxopt.solvers.qp(P, q, G, h, A, b,kktsolver='ldl', options={'kktreg':1e-9});
        self.w = np.zeros(width)
        self.alpha = solution['x'];
        for i in range(size):
            self.w += x[i]*self.alpha[i]*y[i]
        for i in range(size):
            if self.alpha[i]>1e-5 and self.alpha[i]<C-1e-5:
                self.b =1/y[i] - np.dot(self.w,x[i]);
                break;

    def classify(self,point):
        return np.sign(self.kernel(np.array(point),self.w)+self.b)

    def generateRating(self,data):
        total = 0
        correct = 0
        for sample in data:
            total = total + 1
            if sample[0] == self.classify(sample[1:]):
                correct = correct + 1
        return(float(correct)/total)

def multiClassifierRun(train_data,test_data):
    #initialize and train classifier
    classifiers = [];
    for i in range(8):
        # Create pair of classifier and their accuracy/weight
        classifiers.append({'classifier':SupportVectorMachine(),'rating':0})
        print("TRAINING CLASSIFIER " + str(i))
        classifiers[i]['classifier'].trainDualSoft(train_data[i*1000:(i+1)*1000],0.05)
        classifiers[i]['rating'] = classifiers[i]['classifier'].generateRating(train_data[8000:])
        print("CLASSIFIER " + str(i) + " HAS TRUST RATING: " + str(classifiers[i]['rating']))

    # Find total accuracy
    totalAccuracy = 0;
    for i in range(8):
        totalAccuracy = classifiers[i]['rating'] + totalAccuracy
    #Weight Everyone's votes
    for i in range(8):
        classifiers[i]['rating'] = float(classifiers[i]['rating'])/totalAccuracy

    # TEST ALGORITHM
    correct = 0
    total = 0
    for sample in train_data:
        total = total + 1
        result = 0
        for classifierPair in classifiers:
            result = result + (classifierPair['classifier'].classify(sample[1:])*classifierPair['rating'])
        if sample[0] == np.sign(result):
            correct = correct + 1;

    print("MULTI-SVM - TRAINING ACCURACY : " + str(round((float(correct)/total)*100,2)) + '%')

    correct = 0
    total = 0
    for sample in test_data:
        total = total + 1
        result = 0
        for classifierPair in classifiers:
            result = result + (classifierPair['classifier'].classify(sample[1:])*classifierPair['rating'])
        if sample[0] == np.sign(result):
            correct = correct + 1;

    print("MULTI-SVM - TESTING ACCURACY : " + str(round((float(correct)/total)*100,2)) + '%')


def calcwDiff(w1,w2):
    w3 = np.zeros(len(w1))
    for i in range(len(w1)):
        w3[i] = abs(float(w1[i] - w2[i])/w2[i])
    avg = 0;
    for i in w3:
        avg=avg+i
    avg = avg/len(w3)
    return round(avg,4);

train_data = pd.read_csv("data/train.csv",header=None)
train_data = train_data.values
train_data = preprocess_data(train_data)


test_data = pd.read_csv("data/test.csv",header=None)
test_data = test_data.values
test_data = preprocess_data(test_data)

################### COMPARE ON FULL DATASET #########################
# multiClassifierRun(train_data,test_data);
# SciKitClassifier = LinearSVC(C=0.01, class_weight=None, dual=True, fit_intercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#      multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
#      verbose=0)
# x = np.array([row[1:] for row in train_data[0:]])
# y = np.array([row[0] for row in train_data])
# SciKitClassifier.fit(x,y)
# correct = 0
# total = 0
# for i in train_data[:1500]:
#     total=total+1;
#     if(SciKitClassifier.predict([i[1:]])==i[0]):
#         correct=correct+1;
# print("SkLearn: Training accuracy" + str(float(correct)/total))
#
# correct = 0
# total = 0
# for i in test_data:
#     total=total+1;
#     if(SciKitClassifier.predict([i[1:]])==i[0]):
#         correct=correct+1;
# print("SkLearn: Testing accuracy: " + str(float(correct)/total))
#
# exit()
#####################################################################


attributes = []
temp = []
SciKitClassifier = LinearSVC(C=0.01, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)

x = np.array([row[1:] for row in train_data[0:]])
y = np.array([row[0] for row in train_data])
SciKitClassifier.fit(x[:1500],y[:1500])

temp.append(SciKitClassifier.coef_[0])
temp.append(SciKitClassifier.intercept_[0])

correct = 0
total = 0
for i in train_data[:1500]:
    total=total+1;
    if(SciKitClassifier.predict([i[1:]])==i[0]):
        correct=correct+1;
print("Training accuracy" + str(float(correct)/total))
temp.append(float(correct)/total)

correct = 0
total = 0

for i in test_data:
    total=total+1;
    if(SciKitClassifier.predict([i[1:]])==i[0]):
        correct=correct+1;
print("Testing accuracy: " + str(float(correct)/total))

temp.append(float(correct)/total)

attributes.append(temp)

myClass = SupportVectorMachine();

temp = [];
myClass.trainPrimalSoft(train_data[:1500],0.01)
temp.append(myClass.w)
temp.append(myClass.b)
temp.append(myClass.generateRating(train_data[:1500]))
temp.append(myClass.generateRating(test_data))
attributes.append(temp)

temp = [];
myClass.trainPrimalHard(train_data[:1500])
temp.append(myClass.w)
temp.append(myClass.b)
temp.append(myClass.generateRating(train_data[:1500]))
temp.append(myClass.generateRating(test_data))
attributes.append(temp)

temp = [];
myClass.trainDualSoft(train_data[:1500],0.01)
temp.append(myClass.w)
temp.append(myClass.b)
temp.append(myClass.generateRating(train_data[:1500]))
temp.append(myClass.generateRating(test_data))
attributes.append(temp)

temp = [];
myClass.trainDualHard(train_data[:1500])
temp.append(myClass.w)
temp.append(myClass.b)
temp.append(myClass.generateRating(train_data[:1500]))
temp.append(myClass.generateRating(test_data))
attributes.append(temp)


#Print out Accuracy scores:
temp = [];
temp.append("sklearn")
temp.append("Primal Soft")
temp.append("Primal Hard")
temp.append("Dual Soft")
temp.append("Dual Hard")

print("\t\t Training Acc\tTesting Acc")
for i in range(len(attributes)):
    print("\t" + str(attributes[i][2]) + "\t" + str(attributes[i][3]))


# Print out deltas across data:



table = [['' for i in range(5)] for j in range(5)]
for i in range(5):
    for j in range(5):
        table[i][j] = str(calcwDiff(attributes[i][0],attributes[j][0])) + "," + str(round(abs(float(attributes[i][1]-attributes[j][1])/attributes[j][1]),4))

frame = pd.DataFrame(table, temp, temp)
print(frame)

exit()
