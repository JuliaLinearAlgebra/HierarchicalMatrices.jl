# In this example, we write a simple method to speed up the matrix-vector
# product of the Cauchy matrix:
#
# 1/(i-j+0.5), for 1 ≤ i,j ≤ N
#
# and a vector:

function cauchykernel{T}(::Type{T}, x, y)
    T(inv(x-y+T(0.5)))
end

function cauchymatrix{T}(::Type{T}, N::Int)
    ret = zeros(T, N, N)
    for j in 1:N, i in 1:N
        ret[i,j] = cauchykernel(T,i,j)
    end
    ret
end

# We note, firstly, that the Cauchy kernel is an asymptotically smooth function,
# and so a hierarchical decomposition, with bounded off-diagonal numerical rank
# and satisfying the strong admissibility criterion, is justified.

using HierarchicalMatrices

cauchymatrix{T}(::Type{T}, f::Function, b::Int, d::Int) = cauchymatrix(T, f, 1, b, 1, d)

function cauchymatrix{T}(::Type{T}, f::Function, a::Int, b::Int, c::Int, d::Int)
    if (b-a+1) < BLOCKSIZE && (d-c+1) < BLOCKSIZE
        i = (b-a)÷2
        j = (d-c)÷2
        H = HierarchicalMatrix(T, 2, 2)
        H[Block(1), Block(1)] = T[f(T,i,j) for i=a:a+i, j=c:c+j]
        H[Block(1), Block(2)] = T[f(T,i,j) for i=a:a+i, j=c+j+1:d]
        H[Block(2), Block(1)] = T[f(T,i,j) for i=a+i+1:b, j=c:c+j]
        H[Block(2), Block(2)] = T[f(T,i,j) for i=a+i+1:b, j=c+j+1:d]
        H
    else
        i = (b-a)÷2
        j = (d-c)÷2
        H = HierarchicalMatrix(T, 2, 2)
        H[Block(1), Block(1)] = cauchymatrix(T, f, a, a+i, c, c+j)
        H[Block(1), Block(2)] = cauchymatrix1(T, f, a, a+i, c+j+1, d)
        H[Block(2), Block(1)] = cauchymatrix2(T, f, a+i+1, b, c, c+j)
        H[Block(2), Block(2)] = cauchymatrix(T, f, a+i+1, b, c+j+1, d)
        H
    end
end

function cauchymatrix1{T}(::Type{T}, f::Function, a::Int, b::Int, c::Int, d::Int)
    if (b-a+1) < BLOCKSIZE && (d-c+1) < BLOCKSIZE
        i = (b-a)÷2
        j = (d-c)÷2
        H = HierarchicalMatrix(T, 2, 2)
        H[Block(1), Block(1)] = barycentricmatrix(T, f, a, a+i, c, c+j)
        H[Block(1), Block(2)] = barycentricmatrix(T, f, a, a+i, c+j+1, d)
        H[Block(2), Block(1)] = T[f(T,i,j) for i=a+i+1:b, j=c:c+j]
        H[Block(2), Block(2)] = barycentricmatrix(T, f, a+i+1, b, c+j+1, d)
        H
    else
        i = (b-a)÷2
        j = (d-c)÷2
        H = HierarchicalMatrix(T, 2, 2)
        H[Block(1), Block(1)] = barycentricmatrix(T, f, a, a+i, c, c+j)
        H[Block(1), Block(2)] = barycentricmatrix(T, f, a, a+i, c+j+1, d)
        H[Block(2), Block(1)] = cauchymatrix1(T, f, a+i+1, b, c, c+j)
        H[Block(2), Block(2)] = barycentricmatrix(T, f, a+i+1, b, c+j+1, d)
        H
    end
end

function cauchymatrix2{T}(::Type{T}, f::Function, a::Int, b::Int, c::Int, d::Int)
    if (b-a+1) < BLOCKSIZE && (d-c+1) < BLOCKSIZE
        i = (b-a)÷2
        j = (d-c)÷2
        H = HierarchicalMatrix(T, 2, 2)
        H[Block(1), Block(1)] = barycentricmatrix(T, f, a, a+i, c, c+j)
        H[Block(1), Block(2)] = T[f(T,i,j) for i=a:a+i, j=c+j+1:d]
        H[Block(2), Block(1)] = barycentricmatrix(T, f, a+i+1, b, c, c+j)
        H[Block(2), Block(2)] = barycentricmatrix(T, f, a+i+1, b, c+j+1, d)
        H
    else
        i = (b-a)÷2
        j = (d-c)÷2
        H = HierarchicalMatrix(T, 2, 2)
        H[Block(1), Block(1)] = barycentricmatrix(T, f, a, a+i, c, c+j)
        H[Block(1), Block(2)] = cauchymatrix2(T, f, a, a+i, c+j+1, d)
        H[Block(2), Block(1)] = barycentricmatrix(T, f, a+i+1, b, c, c+j)
        H[Block(2), Block(2)] = barycentricmatrix(T, f, a+i+1, b, c+j+1, d)
        H
    end
end


# It is in the large-N asymptotic regime that the hierarchical approach
# demonstrates quasi-linear scaling, whereas normally we would expect quadratic
# scaling (beyond the BLAS'ed sub-sizes).

for N in [1000;10_000]
    x = rand(N)
    println("Cauchy matrix construction at N = $N")
    @time C = cauchymatrix(Float64, cauchykernel, N, N)
    @time C = cauchymatrix(Float64, cauchykernel, N, N)
    @time CF = cauchymatrix(Float64, N)
    @time CF = cauchymatrix(Float64, N)
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
