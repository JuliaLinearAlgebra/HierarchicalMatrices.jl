@hierarchical HierarchicalMatrix LowRankMatrix Matrix

if VERSION < v"0.6.0-dev.1108" # julia PR #18218
    import Base.LinAlg: arithtype
    function (*){T,S}(H::AbstractHierarchicalMatrix{T}, x::AbstractVector{S})
        TS = promote_op(*, arithtype(T), arithtype(S))
        A_mul_B!(zeros(TS, size(H, 1)), H, x)
    end
    function (*){T,S}(H::AbstractHierarchicalMatrix{T}, x::AbstractMatrix{S})
        TS = promote_op(*, arithtype(T), arithtype(S))
        A_mul_B!(zeros(TS, size(H, 1), size(x, 2)), H, x)
    end
else
    import Base.LinAlg: matprod
    function (*){T,S}(H::AbstractHierarchicalMatrix{T}, x::AbstractVector{S})
        TS = promote_op(matprod, T, S)
        A_mul_B!(zeros(TS, size(H, 1)), H, x)
    end
    function (*){T,S}(H::AbstractHierarchicalMatrix{T}, x::AbstractMatrix{S})
        TS = promote_op(matprod, T, S)
        A_mul_B!(zeros(TS, size(H, 1), size(x, 2)), H, x)
    end
end

Base.A_mul_B!(y::AbstractVecOrMat, H::AbstractHierarchicalMatrix, x::AbstractVecOrMat) = A_mul_B!(y, H, x, 1, 1)
A_mul_B!(y::AbstractVecOrMat, H::AbstractHierarchicalMatrix, x::AbstractVecOrMat, istart::Int, jstart::Int) = A_mul_B!(y, H, x, istart, jstart, 1, 1)

Base.scale!(H::AbstractHierarchicalMatrix, b::AbstractVector) = scale!(H, b, 1)
Base.scale!(b::AbstractVector, H::AbstractHierarchicalMatrix) = scale!(b, H, 1)

add_col!(H::AbstractHierarchicalMatrix, u::Vector, j::Int) = add_col!(H, u, 1, j)

@generated function A_mul_B!(y::AbstractVecOrMat, H::HierarchicalMatrix, x::AbstractVecOrMat, istart::Int, jstart::Int, INCX::Int, INCY::Int)
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
                    A_mul_B!(y, getindex(H.$T, m, n), x, istart + p, jstart + q, INCX, INCY)"
    for l in 2:L
        T = fieldname(H, l)
        str *= "
                elseif Hmn == $l
                    A_mul_B!(y, getindex(H.$T, m, n), x, istart + p, jstart + q, INCX, INCY)"
    end
    str *= "
                end
                q += INCX*blocksize(H, 1, n, 2)
            end
            p += INCY*blocksize(H, m, N, 1)
        end
        return y
    end"
    return parse(str)
end

@generated function scale!(H::HierarchicalMatrix, b::AbstractVector, jstart::Int)
    L = length(fieldnames(H))-1
    T = fieldname(H, 1)
    str = "
    begin
        M, N = blocksize(H)
        q = 0
        for n = 1:N
            for m = 1:M
                Hmn = H.assigned[m,n]
                if Hmn == 1
                    scale!(getindex(H.$T, m, n), b, jstart + q)"
    for l in 2:L
        T = fieldname(H, l)
        str *= "
                elseif Hmn == $l
                    scale!(getindex(H.$T, m, n), getindex(H.$T, m, n), b, jstart + q)"
    end
    str *= "
                end
            end
            q += blocksize(H, 1, n, 2)
        end
        return H
    end"
    return parse(str)
end

@generated function scale!(b::AbstractVector, H::HierarchicalMatrix, istart::Int)
    L = length(fieldnames(H))-1
    T = fieldname(H, 1)
    str = "
    begin
        M, N = blocksize(H)
        p = 0
        for m = 1:M
            for n = 1:N
                Hmn = H.assigned[m,n]
                if Hmn == 1
                    scale!(b, getindex(H.$T, m, n), istart + p)"
    for l in 2:L
        T = fieldname(H, l)
        str *= "
                elseif Hmn == $l
                    scale!(getindex(H.$T, m, n), b, getindex(H.$T, m, n), istart + p)"
    end
    str *= "
                end
            end
            p += blocksize(H, m, N, 1)
        end
        return H
    end"
    return parse(str)
end

@generated function add_col!{S}(H::HierarchicalMatrix{S}, u::Vector{S}, istart::Int, j::Int)
    L = length(fieldnames(H))-1
    T = fieldname(H, 1)
    str = "
    begin
        H1 = deepcopy(H)
        M, N = blocksize(H)
        n = 1
        q = blocksize(H, 1, n, 2)
        while q < j
            n += 1
            q += blocksize(H, 1, n, 2)
        end
        q -= blocksize(H, 1, n, 2)
        p = 0
        for m = 1:M
            Hmn = H.assigned[m,n]
            if Hmn == 1
                add_col!(getindex(H.$T, m, n), u, istart + p, j - q)"
    for l in 2:L
        T = fieldname(H, l)
        str *= "
            elseif Hmn == $l
                if isimmutable(getindex(H.$T, m, n))
                    H[Block(m), Block(n)] = add_col(getindex(H.$T, m, n), u, istart + p, j - q)
                else
                    add_col!(getindex(H.$T, m, n), u, istart + p, j - q)
                end"
    end
    str *= "
            end
            p += blocksize(H, m, N, 1)
        end
        return H
    end"
    return parse(str)
end
