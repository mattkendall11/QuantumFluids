import numpy as np
import pytest
from matrix_construction import construct_M1, construct_M2, construct_N, construct_L
from classical_solver import generate_random_A

def test_construct_M1():
    d, k = 2, 3
    A_list = generate_random_A(d, k, seed=42)
    M1 = construct_M1(A_list)
    assert M1.shape == (k*d, k*d)

def test_construct_M2():
    d, k = 2, 3
    M2 = construct_M2(k, d)
    assert M2.shape == ((k+1)*d, k*d)

def test_construct_N():
    d, k, m, p = 2, 2, 1, 2
    A_list = generate_random_A(d, k, seed=1)
    M1 = construct_M1(A_list)
    M2 = construct_M2(k, d)
    N = construct_N(M1, M2, m, p, d)
    assert N.shape == ((m+p+1)*d, (m+p+1)*d)

def test_construct_L():
    d, k, m, p = 2, 2, 1, 2
    A_list = generate_random_A(d, k, seed=1)
    M1 = construct_M1(A_list)
    M2 = construct_M2(k, d)
    N = construct_N(M1, M2, m, p, d)
    L = construct_L(N)
    assert L.shape == N.shape 