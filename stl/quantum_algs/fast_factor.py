"""
Fast factorization module - combines multiple classical algorithms for speed.
"""

import math
import random
from math import gcd
from typing import Optional, Tuple, List

# Pre-computed small primes for trial division
_SMALL_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
    53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
    193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
    269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347,
    349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
    431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499,
    503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593,
    599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661,
    673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757,
    761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853,
    857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
    947, 953, 967, 971, 977, 983, 991, 997
]


def is_probable_prime(n: int, k: int = 10) -> bool:
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29):
        if n % p == 0:
            return n == p
    
    # Write n-1 = d * 2^s
    s = 0
    d = n - 1
    while d % 2 == 0:
        d //= 2
        s += 1
    
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def trial_division(n: int, limit: int = 10000) -> Optional[Tuple[int, int]]:
    """Fast trial division with precomputed primes."""
    for p in _SMALL_PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            return (p, n // p)
    return None


def pollards_rho(n: int, max_iter: int = 100000) -> Optional[Tuple[int, int]]:
    """Pollard's Rho algorithm for factorization."""
    if n % 2 == 0:
        return (2, n // 2)
    if is_probable_prime(n):
        return None
    
    while True:
        c = random.randrange(1, n)
        f = lambda x: (x * x + c) % n
        
        x, y, d = 2, 2, 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        
        if d != n:
            return (d, n // d)


def pollards_p1(n: int, B: int = 10000) -> Optional[Tuple[int, int]]:
    """Pollard's p-1 algorithm - good when p-1 is smooth."""
    if n % 2 == 0:
        return (2, n // 2)
    
    a = 2
    for p in _SMALL_PRIMES:
        if p > B:
            break
        # a = a^(p^k) mod n where p^k is largest power <= B
        e = int(math.log(B) / math.log(p))
        a = pow(a, pow(p, e), n)
        if a == 1:
            continue
        d = math.gcd(a - 1, n)
        if 1 < d < n:
            return (d, n // d)
    return None


def factor(n: int) -> List[int]:
    """
    Fast complete factorization.
    Returns sorted list of prime factors.
    """
    if n < 2:
        return []
    
    factors = []
    
    # Quick trial division
    for p in _SMALL_PRIMES:
        while n % p == 0:
            factors.append(p)
            n //= p
        if n == 1:
            return factors
        if p * p > n:
            factors.append(n)
            return factors
    
    # If remaining is prime
    if is_probable_prime(n):
        factors.append(n)
        return sorted(factors)
    
    # Pollard's Rho for remaining composite
    def _factor(n: int) -> List[int]:
        if n == 1:
            return []
        if is_probable_prime(n):
            return [n]
        
        # Try Pollard's p-1 first (fast for smooth p-1)
        fac = pollards_p1(n)
        if fac:
            a, b = fac
            return _factor(a) + _factor(b)
        
        # Pollard's Rho
        fac = pollards_rho(n)
        if fac:
            a, b = fac
            return _factor(a) + _factor(b)
        
        # Fallback: trial division up to sqrt
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return _factor(i) + _factor(n // i)
        return [n]  # should not reach here
    
    factors.extend(_factor(n))
    return sorted(factors)


def factor_pair(n: int) -> Tuple[int, int]:
    """Return any non-trivial factor pair, or (n, 1) if prime."""
    if n < 2:
        return (n, 1)
    
    # Quick checks
    if n % 2 == 0:
        return (2, n // 2)
    for p in _SMALL_PRIMES[:100]:
        if n % p == 0:
            return (p, n // p)
    
    if is_probable_prime(n):
        return (n, 1)
    
    # Try fast methods
    for method in [pollards_p1, pollards_rho]:
        fac = method(n)
        if fac:
            return fac
    
    return (n, 1)  # Prime


def is_prime(n: int) -> bool:
    """Fast primality test."""
    return is_probable_prime(n, k=20)


def factorize(n: int) -> List[int]:
    """Alias for factor()."""
    return factor(n)