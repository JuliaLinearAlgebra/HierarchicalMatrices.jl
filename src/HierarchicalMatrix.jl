@hierarchical HierarchicalMatrix LowRankMatrix Matrix

function (*){T,S}(H::AbstractHierarchicalMatrix{T}, x::AbstractVector{S})
    TS = promote_op(*, arithtype(T), arithtype(S))
    A_mul_B!(zeros(TS, size(H, 1)), H, x)
end

Base.A_mul_B!(u::Vector, H::AbstractHierarchicalMatrix, v::AbstractVector) = A_mul_B!(u, H, v, 1, 1)

Base.scale!(H::AbstractHierarchicalMatrix, b::AbstractVector) = scale!(H, b, 1)
Base.scale!(b::AbstractVector, H::AbstractHierarchicalMatrix) = scale!(b, H, 1)

add_col!(H::AbstractHierarchicalMatrix, u::Vector, j::Int) = add_col!(H, u, 1, j)

function Base.getindex(H::HierarchicalMatrix, i::Int, j::Int)
    p, q = size(H)
    M, N = blocksize(H)

    m = 1
    while m ≤ M
        r = blocksize(H, m, N, 1)
        if i > r
            i -= r
            m += 1
        else
            break
        end
    end

    n = 1
    while n ≤ N
        s = blocksize(H, 1, n, 2)
        if j > s
            j -= s
            n += 1
        else
            break
        end
    end

    blockgetindex(H, m, n, i, j)
end

@generated function Base.A_mul_B!(u::Vector, H::HierarchicalMatrix, v::AbstractVector, istart::Int, jstart::Int)
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

@generated function Base.scale!(H::HierarchicalMatrix, b::AbstractVector, jstart::Int)
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

@generated function Base.scale!(b::AbstractVector, H::HierarchicalMatrix, istart::Int)
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
