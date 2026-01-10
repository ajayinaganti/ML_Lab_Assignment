import numpy as np
import statistics as st
import random

# 1st code
def countVowelsConsonents(string):
    vowels = "aeiouAEIOU"
    consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
    vowel_count=0
    consonent_count=0
    for i in string:
        if i in vowels:
            vowel_count+=1
        elif i in consonants:
            consonent_count+=1
    return vowel_count, consonent_count

# 2nd code

def MultiplyMatrices(A,B):
    
    if A.shape[1]!=B.shape[0]:
        return f"Error! Cannot be multiplied"
    else:
        C = np.dot(A,B)
        return C

# 3rd code

def CommonElementChecker(List1,List2):
    common_count=0
    Temp_list=List2.copy()
    for x in List1:
        if x in Temp_list:
            common_count+=1
            Temp_list.remove(x)
    return common_count

# 4th code

def MatrixTranspose(Matrix):
    return Matrix.T

# 5th code

def MeanMedianMode(list):
    return f"Mean:{st.mean(list)}, Median:{st.median(list)}, Mode:{st.mode(list)}"




# 1st Question
print("1st Question\nEnter the String: ")

string = str(input())
print(countVowelsConsonents(string))

# 2nd Question

print("2nd Question")
A_rows = int(input("Enter rows of matrix A: "))
A_coloums = int(input("Enter coloumns of matrix A: "))

print("Elements of A")
A = np.array([list(map(int, input().split())) for _ in range(A_rows)])

B_rows = int(input("Enter rows of matrix B: "))
B_coloums = int(input("Enter coloumns of matrix B: "))

print("Enter elements of B")
B = np.array([list(map(int, input().split())) for _ in range(B_rows)])

print(MultiplyMatrices(A,B))


# 3rd Question

print("3rd Question")

l1 = [1,4,8,5,8]
l2 = [4,6,3,8]
print(CommonElementChecker(l1,l2))

# 4th question

print("4th Question")
r = int(input("Enter no. of rows: "))
print("Enter the Elements of the Matrix")
mat = np.array([list(map(int, input().split())) for _ in range(r)])
print(MatrixTranspose(mat))

# 5th Question

print("5th Question")
numbers = []
for i in range(100):
    numbers.append(random.randint(100,150))
 

print(MeanMedianMode(numbers))



