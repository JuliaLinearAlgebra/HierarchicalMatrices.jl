@hierarchical KernelMatrix BarycentricMatrix2D Matrix

if VERSION < v"0.6.0-dev.1108" # julia PR #18218
    import Base.LinAlg: arithtype
    function (*){T,S}(H::AbstractKernelMatrix{T}, x::AbstractVector{S})
        TS = promote_op(*, arithtype(T), arithtype(S))
        A_mul_B!(zeros(TS, size(H, 1)), H, x)
    end
    function (*){T,S}(H::AbstractKernelMatrix{T}, x::AbstractMatrix{S})
        TS = promote_op(*, arithtype(T), arithtype(S))
        A_mul_B!(zeros(TS, size(H, 1), size(x, 2)), H, x)
    end
else
    import Base.LinAlg: matprod
    function (*){T,S}(H::AbstractKernelMatrix{T}, x::AbstractVector{S})
        TS = promote_op(matprod, T, S)
        A_mul_B!(zeros(TS, size(H, 1)), H, x)
    end
    function (*){T,S}(H::AbstractKernelMatrix{T}, x::AbstractMatrix{S})
        TS = promote_op(matprod, T, S)
        A_mul_B!(zeros(TS, size(H, 1), size(x, 2)), H, x)
    end
end

Base.A_mul_B!(u::Vector, H::AbstractKernelMatrix, v::AbstractVector) = A_mul_B!(u, H, v, 1, 1)

@generated function A_mul_B!{S}(u::Vector{S}, H::KernelMatrix{S}, v::AbstractVector{S}, istart::Int, jstart::Int)
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
                    A_mul_B!(u, getindex(H.$T, m, n), v, istart + p, jstart + q)"
    for l in 2:L
        T = fieldname(H, l)
        str *= "
                elseif Hmn == $l
                    A_mul_B!(u, getindex(H.$T, m, n), v, istart + p, jstart + q)"
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

KernelMatrix{T}(f::Function, x::Vector{T}, y::Vector{T}, a::T, b::T, c::T, d::T) = KernelMatrix(f, x, y, 1:length(x), 1:length(y), a, b, c, d)

function KernelMatrix{T}(f::Function, x::Vector{T}, y::Vector{T}, ir::UnitRange{Int}, jr::UnitRange{Int}, a::T, b::T, c::T, d::T)
    ir1, ir2 = indsplit(x, ir, a, b)
    jr1, jr2 = indsplit(y, jr, c, d)
    ab2 = half(T)*(a+b)
    cd2 = half(T)*(c+d)

    if length(ir1) < BLOCKSIZE(T) && length(ir2) < BLOCKSIZE(T) && length(jr1) < BLOCKSIZE(T) && length(jr2) < BLOCKSIZE(T)
        H = KernelMatrix(T, 2, 2)
        H[Block(1), Block(1)] = T[f(x[i], y[j]) for i in ir1, j in jr1]
        H[Block(1), Block(2)] = T[f(x[i], y[j]) for i in ir1, j in jr2]
        H[Block(2), Block(1)] = T[f(x[i], y[j]) for i in ir2, j in jr1]
        H[Block(2), Block(2)] = T[f(x[i], y[j]) for i in ir2, j in jr2]
        H
    else
        H = KernelMatrix(T, 2, 2)
        H[Block(1), Block(1)] = KernelMatrix(f, x, y, ir1, jr1, a, ab2, c, cd2)
        H[Block(1), Block(2)] = KernelMatrix1(f, x, y, ir1, jr2, a, ab2, cd2, d)
        H[Block(2), Block(1)] = KernelMatrix2(f, x, y, ir2, jr1, ab2, b, c, cd2)
        H[Block(2), Block(2)] = KernelMatrix(f, x, y, ir2, jr2, ab2, b, cd2, d)
        H
    end
end

function KernelMatrix1{T}(f::Function, x::Vector{T}, y::Vector{T}, ir::UnitRange{Int}, jr::UnitRange{Int}, a::T, b::T, c::T, d::T)
    ir1, ir2 = indsplit(x, ir, a, b)
    jr1, jr2 = indsplit(y, jr, c, d)
    ab2 = half(T)*(a+b)
    cd2 = half(T)*(c+d)

    if length(ir1) < BLOCKSIZE(T) && length(ir2) < BLOCKSIZE(T) && length(jr1) < BLOCKSIZE(T) && length(jr2) < BLOCKSIZE(T)
        H = KernelMatrix(T, 2, 2)
        H[Block(1), Block(1)] = BarycentricMatrix2D(T, f, a, ab2, c, cd2, x, y, ir1, jr1)
        H[Block(1), Block(2)] = BarycentricMatrix2D(T, f, a, ab2, cd2, d, x, y, ir1, jr2)
        H[Block(2), Block(1)] = T[f(x[i], y[j]) for i in ir2, j in jr1]
        H[Block(2), Block(2)] = BarycentricMatrix2D(T, f, ab2, b, cd2, d, x, y, ir2, jr2)
        H
    else
        H = KernelMatrix(T, 2, 2)
        H[Block(1), Block(1)] = BarycentricMatrix2D(T, f, a, ab2, c, cd2, x, y, ir1, jr1)
        H[Block(1), Block(2)] = BarycentricMatrix2D(T, f, a, ab2, cd2, d, x, y, ir1, jr2)
        H[Block(2), Block(1)] = KernelMatrix1(f, x, y, ir2, jr1, ab2, b, c, cd2)
        H[Block(2), Block(2)] = BarycentricMatrix2D(T, f, ab2, b, cd2, d, x, y, ir2, jr2)
        H
    end
end

function KernelMatrix2{T}(f::Function, x::Vector{T}, y::Vector{T}, ir::UnitRange{Int}, jr::UnitRange{Int}, a::T, b::T, c::T, d::T)
    ir1, ir2 = indsplit(x, ir, a, b)
    jr1, jr2 = indsplit(y, jr, c, d)
    ab2 = half(T)*(a+b)
    cd2 = half(T)*(c+d)

    if length(ir1) < BLOCKSIZE(T) && length(ir2) < BLOCKSIZE(T) && length(jr1) < BLOCKSIZE(T) && length(jr2) < BLOCKSIZE(T)
        H = KernelMatrix(T, 2, 2)
        H[Block(1), Block(1)] = BarycentricMatrix2D(T, f, a, ab2, c, cd2, x, y, ir1, jr1)
        H[Block(1), Block(2)] = T[f(x[i], y[j]) for i in ir1, j in jr2]
        H[Block(2), Block(1)] = BarycentricMatrix2D(T, f, ab2, b, c, cd2, x, y, ir2, jr1)
        H[Block(2), Block(2)] = BarycentricMatrix2D(T, f, ab2, b, cd2, d, x, y, ir2, jr2)
        H
    else
        H = KernelMatrix(T, 2, 2)
        H[Block(1), Block(1)] = BarycentricMatrix2D(T, f, a, ab2, c, cd2, x, y, ir1, jr1)
        H[Block(1), Block(2)] = KernelMatrix2(f, x, y, ir1, jr2, a, ab2, cd2, d)
        H[Block(2), Block(1)] = BarycentricMatrix2D(T, f, ab2, b, c, cd2, x, y, ir2, jr1)
        H[Block(2), Block(2)] = BarycentricMatrix2D(T, f, ab2, b, cd2, d, x, y, ir2, jr2)
        H
    end
end
