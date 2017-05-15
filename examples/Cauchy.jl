# In this example, we write a simple method to speed up the matrix-vector
# product of the Cauchy matrix:
#
# 1/(x_i-y_j), for 1 ≤ i,j ≤ N
#
# and a vector:

function cauchykernel{T}(::Type{T}, x, y)
    T(inv(x-y))
end

function cauchymatrix{T}(::Type{T}, x::Vector{T}, y::Vector{T})
    ret = zeros(T, length(x), length(y))
    for j in 1:length(y), i in 1:length(x)
        ret[i,j] = cauchykernel(T, x[i], y[j])
    end
    ret
end

# We note, firstly, that the Cauchy kernel is an asymptotically smooth function,
# and so a hierarchical decomposition, with bounded off-diagonal numerical rank
# and satisfying the strong admissibility criterion, is justified.

using HierarchicalMatrices

import HierarchicalMatrices: chebyshevpoints, half

@hierarchical CauchyMatrix BarycentricMatrix2D Matrix

import Base: promote_op
import Base: +, -, *, /, \, .+, .-, .*, ./, .\, ==, !=

if VERSION < v"0.6-"
    import Base.LinAlg: arithtype
    function (*){T,S}(H::AbstractCauchyMatrix{T}, x::AbstractVector{S})
        TS = promote_op(*, arithtype(T), arithtype(S))
        A_mul_B!(zeros(TS, size(H, 1)), H, x)
    end
else
    import Base.LinAlg: matprod
    function (*){T,S}(H::AbstractCauchyMatrix{T}, x::AbstractVector{S})
        TS = promote_op(matprod, T, S)
        A_mul_B!(zeros(TS, size(H, 1)), H, x)
    end
end

Base.A_mul_B!(u::Vector, H::AbstractCauchyMatrix, v::AbstractVector) = HierarchicalMatrices.A_mul_B!(u, H, v, 1, 1)

@generated function HierarchicalMatrices.A_mul_B!{S}(u::Vector{S}, H::CauchyMatrix{S}, v::AbstractVector{S}, istart::Int, jstart::Int)
    L = length(fieldnames(H))-1
    T = fieldname(H, 1)
    str = "
    begin
        M, N = blocksize(H)
        p = 0
        for m = 1:M
            q = 0
            for n = 1:N
                Hmn = H.assigned[m,n]
                if Hmn == 1
                    HierarchicalMatrices.A_mul_B!(u, getindex(H.$T, m, n), v, istart + p, jstart + q)"
    for l in 2:L
        T = fieldname(H, l)
        str *= "
                elseif Hmn == $l
                    HierarchicalMatrices.A_mul_B!(u, getindex(H.$T, m, n), v, istart + p, jstart + q)"
    end
    str *= "
                end
                q += blocksize(H, 1, n, 2)
            end
            p += blocksize(H, m, N, 1)
        end
        return u
    end"
    return parse(str)
end

CauchyMatrix{T}(x::Vector{T}, y::Vector{T}, a::T, b::T, c::T, d::T) = CauchyMatrix(x, y, 1:length(x), 1:length(y), a, b, c, d)

function CauchyMatrix{T}(x::Vector{T}, y::Vector{T}, ir::UnitRange{Int}, jr::UnitRange{Int}, a::T, b::T, c::T, d::T)
    ir1, ir2 = indsplit(x, ir, a, b)
    jr1, jr2 = indsplit(y, jr, c, d)
    ab2 = half(T)*(a+b)
    cd2 = half(T)*(c+d)

    if length(ir1) < BLOCKSIZE(T) && length(ir2) < BLOCKSIZE(T) && length(jr1) < BLOCKSIZE(T) && length(jr2) < BLOCKSIZE(T)
        H = CauchyMatrix(T, 2, 2)
        H[Block(1), Block(1)] = T[cauchykernel(T,x[i],y[j]) for i in ir1, j in jr1]
        H[Block(1), Block(2)] = T[cauchykernel(T,x[i],y[j]) for i in ir1, j in jr2]
        H[Block(2), Block(1)] = T[cauchykernel(T,x[i],y[j]) for i in ir2, j in jr1]
        H[Block(2), Block(2)] = T[cauchykernel(T,x[i],y[j]) for i in ir2, j in jr2]
        H
    else
        H = CauchyMatrix(T, 2, 2)
        H[Block(1), Block(1)] = CauchyMatrix(x, y, ir1, jr1, a, ab2, c, cd2)
        H[Block(1), Block(2)] = CauchyMatrix1(x, y, ir1, jr2, a, ab2, cd2, d)
        H[Block(2), Block(1)] = CauchyMatrix2(x, y, ir2, jr1, ab2, b, c, cd2)
        H[Block(2), Block(2)] = CauchyMatrix(x, y, ir2, jr2, ab2, b, cd2, d)
        H
    end
end

function CauchyMatrix1{T}(x::Vector{T}, y::Vector{T}, ir::UnitRange{Int}, jr::UnitRange{Int}, a::T, b::T, c::T, d::T)
    ir1, ir2 = indsplit(x, ir, a, b)
    jr1, jr2 = indsplit(y, jr, c, d)
    ab2 = half(T)*(a+b)
    cd2 = half(T)*(c+d)

    if length(ir1) < BLOCKSIZE(T) && length(ir2) < BLOCKSIZE(T) && length(jr1) < BLOCKSIZE(T) && length(jr2) < BLOCKSIZE(T)
        H = CauchyMatrix(T, 2, 2)
        H[Block(1), Block(1)] = BarycentricMatrix2D(T, cauchykernel, a, ab2, c, cd2, x, y, ir1, jr1)
        H[Block(1), Block(2)] = BarycentricMatrix2D(T, cauchykernel, a, ab2, cd2, d, x, y, ir1, jr2)
        H[Block(2), Block(1)] = T[cauchykernel(T,x[i],y[j]) for i in ir2, j in jr1]
        H[Block(2), Block(2)] = BarycentricMatrix2D(T, cauchykernel, ab2, b, cd2, d, x, y, ir2, jr2)
        H
    else
        H = CauchyMatrix(T, 2, 2)
        H[Block(1), Block(1)] = BarycentricMatrix2D(T, cauchykernel, a, ab2, c, cd2, x, y, ir1, jr1)
        H[Block(1), Block(2)] = BarycentricMatrix2D(T, cauchykernel, a, ab2, cd2, d, x, y, ir1, jr2)
        H[Block(2), Block(1)] = CauchyMatrix1(x, y, ir2, jr1, ab2, b, c, cd2)
        H[Block(2), Block(2)] = BarycentricMatrix2D(T, cauchykernel, ab2, b, cd2, d, x, y, ir2, jr2)
        H
    end
end

function CauchyMatrix2{T}(x::Vector{T}, y::Vector{T}, ir::UnitRange{Int}, jr::UnitRange{Int}, a::T, b::T, c::T, d::T)
    ir1, ir2 = indsplit(x, ir, a, b)
    jr1, jr2 = indsplit(y, jr, c, d)
    ab2 = half(T)*(a+b)
    cd2 = half(T)*(c+d)

    if length(ir1) < BLOCKSIZE(T) && length(ir2) < BLOCKSIZE(T) && length(jr1) < BLOCKSIZE(T) && length(jr2) < BLOCKSIZE(T)
        H = CauchyMatrix(T, 2, 2)
        H[Block(1), Block(1)] = BarycentricMatrix2D(T, cauchykernel, a, ab2, c, cd2, x, y, ir1, jr1)
        H[Block(1), Block(2)] = T[cauchykernel(T,x[i],y[j]) for i in ir1, j in jr2]
        H[Block(2), Block(1)] = BarycentricMatrix2D(T, cauchykernel, ab2, b, c, cd2, x, y, ir2, jr1)
        H[Block(2), Block(2)] = BarycentricMatrix2D(T, cauchykernel, ab2, b, cd2, d, x, y, ir2, jr2)
        H
    else
        H = CauchyMatrix(T, 2, 2)
        H[Block(1), Block(1)] = BarycentricMatrix2D(T, cauchykernel, a, ab2, c, cd2, x, y, ir1, jr1)
        H[Block(1), Block(2)] = CauchyMatrix2(x, y, ir1, jr2, a, ab2, cd2, d)
        H[Block(2), Block(1)] = BarycentricMatrix2D(T, cauchykernel, ab2, b, c, cd2, x, y, ir2, jr1)
        H[Block(2), Block(2)] = BarycentricMatrix2D(T, cauchykernel, ab2, b, cd2, d, x, y, ir2, jr2)
        H
    end
end


# It is in the large-N asymptotic regime that the hierarchical approach
# demonstrates quasi-linear scaling, whereas normally we would expect quadratic
# scaling (beyond the BLAS'ed sub-sizes).

for N in [1000;10_000]
    b = rand(N)
    x = chebyshevpoints(Float64, N)
    y = chebyshevpoints(Float64, N; kind = 2)
    println("Cauchy matrix construction at N = $N")
    @time C = CauchyMatrix(x, y, 1.0, -1.0, 1.0, -1.0)
    @time C = CauchyMatrix(x, y, 1.0, -1.0, 1.0, -1.0)
    @time CF = cauchymatrix(Float64, x, y)
    @time CF = cauchymatrix(Float64, x, y)
    println()
    println("Cauchy matrix-vector multiplication at N = $N")
    @time C*b
    @time C*b
    @time CF*b
    @time CF*b
    println()
    println("Relative error: ",norm(C*b - CF*b)/norm(C*b))
    println()
    println()
end
