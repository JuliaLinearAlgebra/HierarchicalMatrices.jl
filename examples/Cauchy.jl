# In this example, we write a simple method to speed up the matrix-vector
# product of the Cauchy matrix:
#
# 1/(i-j+0.5), for 1 ≤ i,j ≤ n
#
# and a vector:

function cauchykernel{T}(::Type{T}, x, y)
    T(inv(x-y+T(0.5)))
end

# We note, firstly, that the Cauchy kernel is an asymptotically smooth function,
# and so a hierarchical decomposition, with bounded off-diagonal numerical rank
# and satisfying the strong admissibility criterion, is justified.

using HierarchicalMatrices

@hierarchicalmatrix Cauchy BarycentricMatrix Matrix

Cauchy{T}(::Type{T}, f::Function, b::Int, d::Int) = Cauchy(T, f, 1, b, 1, d)

function Cauchy{T}(::Type{T}, f::Function, a::Int, b::Int, c::Int, d::Int)
    if (b-a+1) < BLOCKSIZE && (d-c+1) < BLOCKSIZE
        i = (b-a)÷2
        j = (d-c)÷2
        C = Cauchy(T, 2, 2)
        setblock!(C, T[f(T,i,j) for i=a:a+i, j=c:c+j], 1, 1)
        setblock!(C, T[f(T,i,j) for i=a:a+i, j=c+j+1:d], 1, 2)
        setblock!(C, T[f(T,i,j) for i=a+i+1:b, j=c:c+j], 2, 1)
        setblock!(C, T[f(T,i,j) for i=a+i+1:b, j=c+j+1:d], 2, 2)
        C
    else
        i = (b-a)÷2
        j = (d-c)÷2
        C = Cauchy(T, 2, 2)
        setblock!(C, Cauchy(T, f, a, a+i, c, c+j), 1, 1)
        setblock!(C, cauchy1(T, f, a, a+i, c+j+1, d), 1, 2)
        setblock!(C, cauchy2(T, f, a+i+1, b, c, c+j), 2, 1)
        setblock!(C, Cauchy(T, f, a+i+1, b, c+j+1, d), 2, 2)
        C
    end
end

function cauchy1{T}(::Type{T}, f::Function, a::Int, b::Int, c::Int, d::Int)
    if (b-a+1) < BLOCKSIZE && (d-c+1) < BLOCKSIZE
        i = (b-a)÷2
        j = (d-c)÷2
        C = Cauchy(T, 2, 2)
        setblock!(C, BarycentricMatrix(T, f, a, a+i, c, c+j), 1, 1)
        setblock!(C, BarycentricMatrix(T, f, a, a+i, c+j+1, d), 1, 2)
        setblock!(C, T[f(T,i,j) for i=a+i+1:b, j=c:c+j], 2, 1)
        setblock!(C, BarycentricMatrix(T, f, a+i+1, b, c+j+1, d), 2, 2)
        C
    else
        i = (b-a)÷2
        j = (d-c)÷2
        C = Cauchy(T, 2, 2)
        setblock!(C, BarycentricMatrix(T, f, a, a+i, c, c+j), 1, 1)
        setblock!(C, BarycentricMatrix(T, f, a, a+i, c+j+1, d), 1, 2)
        setblock!(C, cauchy1(T, f, a+i+1, b, c, c+j), 2, 1)
        setblock!(C, BarycentricMatrix(T, f, a+i+1, b, c+j+1, d), 2, 2)
        C
    end
end

function cauchy2{T}(::Type{T}, f::Function, a::Int, b::Int, c::Int, d::Int)
    if (b-a+1) < BLOCKSIZE && (d-c+1) < BLOCKSIZE
        i = (b-a)÷2
        j = (d-c)÷2
        C = Cauchy(T, 2, 2)
        setblock!(C, BarycentricMatrix(T, f, a, a+i, c, c+j), 1, 1)
        setblock!(C, T[f(T,i,j) for i=a:a+i, j=c+j+1:d], 1, 2)
        setblock!(C, BarycentricMatrix(T, f, a+i+1, b, c, c+j), 2, 1)
        setblock!(C, BarycentricMatrix(T, f, a+i+1, b, c+j+1, d), 2, 2)
        C
    else
        i = (b-a)÷2
        j = (d-c)÷2
        C = Cauchy(T, 2, 2)
        setblock!(C, BarycentricMatrix(T, f, a, a+i, c, c+j), 1, 1)
        setblock!(C, cauchy2(T, f, a, a+i, c+j+1, d), 1, 2)
        setblock!(C, BarycentricMatrix(T, f, a+i+1, b, c, c+j), 2, 1)
        setblock!(C, BarycentricMatrix(T, f, a+i+1, b, c+j+1, d), 2, 2)
        C
    end
end

# It is in the large-N asymptotic regime that the hierarchical approach
# demonstrates quasi-linear scaling, whereas normally we would expect quadratic
# scaling (beyond the BLAS'ed sub-sizes).

for N in [1000;10_000]
    x = rand(N)
    println("Cauchy matrix construction at N = $N")
    @time C = Cauchy(Float64, cauchykernel, N, N)
    @time C = Cauchy(Float64, cauchykernel, N, N)
    @time CF = [cauchykernel(Float64, i, j) for i in 1:N, j in 1:N]
    @time CF = [cauchykernel(Float64, i, j) for i in 1:N, j in 1:N]
    println()
    println("Cauchy matrix-vector multiplication at N = $N")
    @time C*x
    @time C*x
    @time CF*x
    @time CF*x
    println()
    println("Relative error: ",norm(C*x - CF*x)/norm(C*x))
    println()
    println()
end
