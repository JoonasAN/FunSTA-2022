import csv
from pprint import pp, pprint
import numpy as np
import pandas
from sympy import Eq, symbols, solve
import sys
"""A cheap IMU sensor must be calibrated to provide more accurate
readings. In the file “measurements.csv”, there are raw three-axis accelerometer
measurements from the sensor. You also have access to the ground truth of the
acceleration readings, from a “perfect” IMU, which are in the file “groundtruth.csv”.
You may use your preferred coding language (for example, MATLAB, Python) and
necessary libraries to solve this problem"""




def main():
    
    data_R = pandas.read_csv("measurements.csv", header=None)
    data_S = pandas.read_csv("groundtruth.csv", header=None)
    
    data_R = data_R.to_numpy(dtype=object)
    data_S = data_S.to_numpy(dtype=object)
    
    """(a) Consider an affine correction function ~y = A~r +~b that maps a measurement ~r
    to a more accurate reading ~y, in which A is a 3-by-3 matrix and ~b is a 3-by-1
    vector. Use least squares minimization to find A and ~b that would provide the
    best possible correction function for the given data."""

    # Do matrix least squares minimization   σ = Mβ   ->   ^β = (Mt * M)^-1 * Mt * σ
    #      Mt = transpose(M)

    # Make matrix M 
    m = makeMatrixM(data_R)
    

    # get the transpose of M
    mt = np.transpose(m)

    # Create vector sigma from all groundtruth values and transpose to vertical vector
    sigma = np.asmatrix(data_S).flatten()
    #print(sigma)
    sigma = np.transpose(sigma)

    # calculate beta (β) 
    beta = ((mt*m)**-1)*mt*sigma

    """This is Answer to (a)"""
    # Create Matrix A' and offset vector b'
    matrix_A = makeMatrixA(beta)
    print("\nMatrix A: \n",matrix_A)

    b = makeVectorB(beta)
    print("\nOffset vector :\n",b)



    """
    (b) What is the resulting sum-of-squares error after applying the correction
    function to the measurements? Is it zero? If not, why?
    """
    yList = correctionFunction(matrix_A, b, data_R)

    """Answer to (b)"""
    # calculate sum of square errors
    E = calculateSumSqrE(data_S, yList)
    
    print("\nSum of square error = ",E)
    print("\nSum is not 0 because there is a little difference between groundtruth and measurements\n \
because of environmental factors such as temperature, air pressure, humidity and forces on sensor\n - not only accelerometer axis' \n")
    return 0

def makeMatrixM(R):
    # make matrix  M according to lecture 
    # by getting one vector from measurements data (R/result) at a time
    m = []
    for i in range(len(R)):
        r = R[i]
        r0 = np.append(r,[1])   # add 1 as fourth value in measurement vector
        r1 = np.append(r0,[[0,0,0,0],[0,0,0,0]])    # set zeros
        r2 = np.append([0,0,0,0],[r0,[0,0,0,0]])
        r3 = np.append([0,0,0,0],[[0,0,0,0],r0])
        m.append(r1)
        m.append(r2)
        m.append(r3)
    # create as matrix M and get transpose Mt
    m = np.asmatrix(m, dtype='float')
    # print("M shape = ",m.shape)
    # print(m)
    return m

def makeMatrixA(beta):
    # Now we start making matrix A by splitting beta (β) to 3 matrices (rows for A)
    # and removing b from vector. (A is 3x3 matrix, not 4x3)
    a = []
    b = []

    for matrix in np.split(beta,3):
        b.append(matrix.item(3))
        a.append(np.delete(matrix,3,0).flatten().tolist()[0])

    # make the matrix A and vector b. Transpose b to be vertical vector
    matrix_A = np.asmatrix(a,dtype="float")
    b = np.asmatrix(b,dtype="float").T

    return matrix_A

def makeVectorB(beta):
    b = []
    for matrix in np.split(beta,3):
        b.append(matrix.item(3))
    b = np.asmatrix(b,dtype="float").T
    return b

def correctionFunction(A,b,R):
    """Calculate y-vectors with correction function y = A*r+b
        y : [vector]
        A : [matrix]
        b : [vector]
        r : [vector]
    """
    y_List = [] 
    for index in range(len(R)):
        # get r and s from data and make them matrices and transpose to vertical vector
        r = np.asmatrix(R[index]).T
        # calculate y vector
        y = A*r+b
        y_List.append(y)
    return y_List

def calculateSumSqrE(S,yList):
    """Calculate square error"""
    ssqeList = []
    ssqe = 0
    for index in range(len(S)):
        s = np.asmatrix(S[index]).T
        y = yList[index]
        # calculate signed error
        e = s - y     
        # square of one vector = ( eix^2, eiy^2, eiy^z )      
        sqe = np.square(e)  
        # sum of square error = eix^2 +eiy^2+eiy^z of that vector ( float )
        ssqe=sqe.sum()     
        # append sum value to list
        ssqeList.append(ssqe)
    #pprint(ssqeList[0])
    #pprint(sum(ssqeList))
    # return sum of all 
    return sum(ssqeList)


if "__name__==__main__":
    main()
    

