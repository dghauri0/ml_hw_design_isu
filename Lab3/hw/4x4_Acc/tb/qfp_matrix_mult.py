#!/usr/bin/python3

# Program to multiply two Qn.n matrices using nested loops
# A x B + C

# Only supports semetric Q Fixed point values (Qn.n)
N = 16  # Qn.n = Q16.16 (n=16)

# Print decimal value
print_decimal = True 

P = [[0x0000_1111, 0x00000_2222, 0x0000_3333, 0x0000_4444],
    [0x1111_1111, 0x1111_2222, 0x1111_3333, 0x1111_4444],
    [0x2222_1111, 0x2222_2222, 0x2222_3333, 0x2222_4444],
    [0x3333_1111, 0x3333_2222, 0x3333_3333, 0x3333_4444]]

Q = [[0x0000_5555, 0x00000_6666, 0x0000_7777, 0x0000_8888],
    [0x5555_5555, 0x5555_6666, 0x5555_7777, 0x5555_8888],
    [0x6666_5555, 0x6666_6666, 0x6666_7777, 0x6666_8888],
    [0x7777_5555, 0x7777_6666, 0x7777_7777, 0x7777_8888]]

R = [[0x000A_000A, 0x000A_000B, 0x000A_000C, 0x000A_000D],
    [0x000A_000A, 0x000A_000B, 0x000A_000C, 0x000A_000D],
    [0x000A_000A, 0x000A_000B, 0x000A_000C, 0x000A_000D],
    [0x000A_000A, 0x000A_000B, 0x000A_000C, 0x000A_000D]]

S = [[0x0, 0x0, 0x0, 0x0],
    [0x0, 0x0, 0x0, 0x0],
    [0x0, 0x0, 0x0, 0x0],
    [0x0, 0x0, 0x0, 0x0]]


# C = (A x B) + C
def multiply(A, B, C):
   # iterate through rows of A
   for i in range(len(A)):
      # iterate through columns of B
      for j in range(len(B[0])):
          # iterate through rows of B
          for k in range(len(B)):
              C[i][j] += ((A[i][k] * B[k][j]) >> N) & (2**(N*2) - 1)


# C = A + B
def add(A, B, C):
   # iterate through rows of A
   for i in range(len(A)):
      # iterate through columns of A
      for j in range(len(A[0])):
         C[i][j] = A[i][j]


# Print the provided matrix values in hex or decimal
def matrix_print(M):
   for x in M:
      out = "["
      for y in x:
         if print_decimal:
            out += str(((y>>N) + (y & (2**N-1))*(2**(-N))))
         else:
            out += hex(y)
         out += ", "
      print(out + "]")


# Print Initial Values
print("P")
matrix_print(P)
print("\nQ")
matrix_print(Q)
print("\nR")
matrix_print(R)
print("\nS")
matrix_print(S)

# Multiply
multiply(P, Q, S)
print("\nMultiply P x Q:\n")
matrix_print(S)

# Multiply with inital
multiply(P, Q, R)
print("\nMultiply P x Q:\n")
matrix_print(R)

# Multiply with inital
add(P, Q, S)
print("\nAdd P + Q:\n")
matrix_print(S)
