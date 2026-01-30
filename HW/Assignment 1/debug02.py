#%%
import numpy as np


#%% 
a = np.array([1, 2, 3])
b = np.array([[1], [2]])
result = a + b

print(result)


#TODO answer the following questions as comments below:
# What is the shape of a? (3,)
# What is the shape of b? (2, 1)
# What is the shape of result? (2, 3)
# Why did this happen? and what occurred? 
# Because numpy is pretty smart, it saw that it's possible to 
# broadcast each row of b (since each row is only 1 element) onto a.
# As a result, it created two rows, the first of which broadcast +1 onto a
# and the second of which broadcast +2 onto a. 


#%% 
a = np.array([1, 2, 3])
b = np.array([4])
result = np.dot(a, b)
print(result)

#TODO answer the following questions as comments below:
# Is the output what you expected? Can you explain how it happened?
# Yes it is what I expected. A dot product requires two 1-D vectors of the sam length.
# Since the lengths are mismatched, it throws an error.


#%% 
a = np.ones((2, 3))
b = np.ones((3, 2))
result = a + b.T 
print(result)

# TODO answer the following questions as comments below:
# Why doesn't this work? If the 2D arrays are meant to be the same size, fix 
# the code so that you can successfully add them. 
# Since none of the dimensions are 1, it is not possible to broadcast.
# As a result, the array dimensions (2, 3) and (3,2) are mismatched and it is unclear how to add the arrays.

#%% 

a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])
result = a + b

#TODO - check the output of this addition, is it what you expected? Answer the 
# following questions as comments below:
#  
#  What is the shape of a and the shape of b? (3,) and (3,1)
#  Why is the result 3x3? and what does the addition do in this case?
# Numpy sees that b has a dimension of 1, and so it's possible to broadcast despite the
# mismatched dimensions. Therefore it brodcasts a .+ 1 as the first row, a[1] .+ 2
# as the second crow, and a .+ 3 as the third row.



#%%
a = np.array([1, 2, 3])
b = np.array(np.eye(3))
result1 = b*a
result2 = np.dot(b,a)
result3 = b@a

# which of these is the intended result? 
# TODO put your answer here as a comment and justify it. 
# It depends on what you're trying to do. If you're trying to 
# edit the identity matrix to have a diogonal of values increasing by 1,
# then result1 is correct (broadcasts each element of a .* each row of b).
# result2 and result3 behave the same for this case (dot() and @ are just matrix multiplication).

#%%
a_column = a.reshape((3,1)) 
result1 = b*a_column
result2 = np.dot(b,a_column)
result3 = b@a_column 

# which of these is the intended result? 
# TODO put your answer here as a comment and justify it. 
# It depends on what you're trying to do. result1 behaves the same
# as in the previous example since a_column and b still have a matching dimension and a has one of its dimensions =1,
# so it broadcasts. result2 and 3 both just perform matrix multiplication,
# resulting in m = #columns of a and n = #rows of b.


#%%
result1 = a_column.T*a # gives [1^2, 2^2, 3^2]
result2 = a_column*a # broadcast each element of a_column .* a, results in 3x3 matrix
result3 = a*a_column # same as result2
# result4 = np.dot(a_column,a)
# result5 = np.dot(a_column,a_column)
result6 = np.dot(a_column.T, a_column)
result7 = a_column.T@a_column

#TODO: answer these questions as comments below: 
#  Why does np.dot(a, a) work? but np.dot(a_column, a_column) doesn't? 
# Because a has dimension (3,), it is more flexible. It's not a row or column vector.
# Therefore numpy just reshapes them as (3,1) and (1,3) internally so that the dot product is valid.
#  If I'm trying to do find the sum of the squares of the elements of a, which of these should I use?
# result6 and result7 both work. 






# if you still have questions about broadcasting, and array dimensions, please see the following:
# https://numpy.org/doc/stable/user/basics.broadcasting.html 
# https://www.youtube.com/watch?v=oG1t3qlzq14 
# %%
