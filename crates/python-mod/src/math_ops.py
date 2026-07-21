# Simple math operations
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Constants
PI = 3.14159