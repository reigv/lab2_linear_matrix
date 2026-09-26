import numpy

print('PART I --------------------------------\n')

print('Example - A 3X3 linear system')

# Create A and b matrices
A = numpy.array([[2, -6, 6], [2, 3, -1], [4, -3, -1]])
b = numpy.array([-8, 15, 19])

# solve for x in Ax = b
x = numpy.linalg.solve(A, b)
print(x)
print('--------------------------------')


print('Part I - Example - Inconsistent and dependent system')

print('Inconsistent system --> No solution')
# inconsistent system
A = numpy.array([[2, 1], [6, 3]])
b = numpy.array([3, 3])

try:
    print(numpy.linalg.solve(A, b))
except numpy.linalg.LinAlgError as error:
    print('Error:', error)

print('Dependent system --> Infinite solutions')
# dependent system
A = numpy.array([[2, 1], [6, 3]])
b = numpy.array([1, 3])

try:
    print(numpy.linalg.solve(A, b))
except numpy.linalg.LinAlgError as error:
    print('Error:', error)

# Check the determinant of A
print('Determinant of A:', numpy.linalg.det(A))

# THERE IS NO SOLUTION FOR INCONSISTENT SYSTEM AND INFINITE SOLUTIONS FOR DEPENDENT SYSTEM
print('--------------------------------')


print('Part I - a 10x10 linear system')

numpy.random.seed(1)

# Create A and b matrices with random
A = 10*numpy.random.rand(10, 10)-5
b = 10*numpy.random.rand(10)-5

# Solve Ax = b
solution = numpy.linalg.solve(A, b)
print(solution)

# To verify the solution works, show Ax - b is near 0
print(sum(abs(numpy.dot(A, solution) - b)))
print('--------------------------------')


print('Part II ------------------------')

# Create A and b matrices
A = numpy.array([[2, 1, 3], [1, -1, 2], [4, 3, 5]])
b = numpy.array([-9, -7, -15])

# (a) Check consistency using the determinant
flag = numpy.linalg.det(A)
print(flag)

if flag == 0:
    print('The system is inconsistent')
else:
    print('The system is consistent')

# (b) Solve Ax = b
if flag != 0:
    print('x = ')
    print(numpy.linalg.solve(A, b))
print('--------------------------------')