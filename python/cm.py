n = 15
k = 15

def S(n, k):
    P = 2*k + 1
    M = 6*k + 1
    return P**(n-1) * ((n+3)*P + n * 2**(n-1) * M)

def A(n, k):
    def f(i):
        return (n+1)*i**n + n*(3*i-1)*(2*i+1)**(n-1) + (i+1)**n
    return 1 + sum(f(i) for i in range(1, k)) + n * k**n

def CM(n, k):
    return S(n, k) + A(n, k)

print(f"CM({n},{k}) = {CM(n, k)}")




# 16_929_109_944_024_661_913_431_311_618