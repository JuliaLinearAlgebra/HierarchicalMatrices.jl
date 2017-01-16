abstract AbstractHierarchicalMatrix{T} <: AbstractMatrix{T}

blocksize(H::AbstractHierarchicalMatrix) = size(H.assigned)

function size(H::AbstractHierarchicalMatrix)
    M, N = blocksize(H)

    p = 0
    for m = 1:M
        p += blockgetsize(H, m, N, 1)
    end

    q = 0
    for n = 1:N
        q += blockgetsize(H, 1, n, 2)
    end

    p, q
end

function (*){T,S}(H::AbstractHierarchicalMatrix{T}, x::AbstractVector{S})
    TS = promote_op(*, arithtype(T), arithtype(S))
    A_mul_B!(zeros(TS, size(H, 1)), H, x)
end

A_mul_B!(u::Vector, H::AbstractHierarchicalMatrix, v::AbstractVector) = A_mul_B!(u, H, v, 1, 1)

macro hierarchicalmatrix(HierarchicalMatrix, matrices...)
    blocks = :(begin end)
    push!(blocks.args, :($(Symbol("$(HierarchicalMatrix)blocks"))::Matrix{$HierarchicalMatrix{T}}))
    for i in 1:length(matrices)
        push!(blocks.args, :($(Symbol("$(matrices[i])blocks"))::Matrix{$(matrices[i]){T}}))
    end
    return esc(quote
        export $HierarchicalMatrix

        immutable $HierarchicalMatrix{T} <: AbstractHierarchicalMatrix{T}
            $blocks
            assigned::Matrix{Int}
        end

        @generated function $HierarchicalMatrix{T}(::Type{T}, M::Int, N::Int)
            L = length(fieldnames($HierarchicalMatrix))
            HM = $HierarchicalMatrix
            ex = :(begin end)
            push!(ex.args, :(data = Vector{Any}($L)))
            push!(ex.args, :(data[1] = Matrix{$HM{T}}(M, N)))
            for l in 2:L-1
                S = $matrices[l-1]
                push!(ex.args, :(data[$l] = Matrix{$S{T}}(M, N)))
            end
            push!(ex.args, :(data[$L] = zeros(Int, M, N)))
            push!(ex.args, :($HM(data...)))
            return ex
        end
        $HierarchicalMatrix(M::Int, N::Int) = $HierarchicalMatrix(Float64, M, N)

        @generated function blockgetsize(H::$HierarchicalMatrix, m::Int, n::Int, k::Int)
            L = length(fieldnames(H))-1
            T = fieldname(H, 1)
            str = "
            begin
                Hmn = H.assigned[m,n]
                if Hmn == 1
                    return size(getindex(H.$T,m,n),k)"
            for l in 2:L
                T = fieldname(H, l)
                str *= "
                elseif Hmn == $l
                    return size(getindex(H.$T,m,n),k)"
            end
            str *="
                end
                return 0
            end"
            return parse(str)
        end

        @generated function blockgetindex{S}(H::$HierarchicalMatrix{S}, m::Int, n::Int, i::Int, j::Int)
            L = length(fieldnames(H))-1
            T = fieldname(H, 1)
            str = "
            begin
                Hmn = H.assigned[m,n]
                if Hmn == 1
                    return getindex(getindex(H.$T,m,n),i,j)"
            for l in 2:L
                T = fieldname(H, l)
                str *= "
                elseif Hmn == $l
                    return getindex(getindex(H.$T,m,n),i,j)"
            end
            str *="
                end
                return zero(S)
            end"
            return parse(str)
        end

        @generated function setblock!{S}(H::$HierarchicalMatrix{S}, A::AbstractMatrix{S}, m::Int, n::Int)
            L = length(fieldnames(H))-1
            T = fieldname(H, 1)
            str = "
            begin
                if typeof(H.$T) == Matrix{typeof(A)}
                    setindex!(H.$T, A, m, n)
                    H.assigned[m,n] = 1
                    return H"
            for l in 2:L
                T = fieldname(H, l)
                str *= "
                elseif typeof(H.$T) == Matrix{typeof(A)}
                    setindex!(H.$T, A, m, n)
                    H.assigned[m,n] = $l
                    return H"
            end
            str *= "
                end
                return H
            end"
            return parse(str)
        end

        @generated function Base.A_mul_B!(u::Vector, H::$HierarchicalMatrix, v::AbstractVector, istart::Int, jstart::Int)
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
                        q += blockgetsize(H, 1, n, 2)
                    end
                    p += blockgetsize(H, m, N, 1)
                end
                return u
            end"
            return parse(str)
        end

        function Base.size(H::$HierarchicalMatrix)
            M, N = blocksize(H)

            p = 0
            for m = 1:M
                p += blockgetsize(H, m, N, 1)
            end

            q = 0
            for n = 1:N
                q += blockgetsize(H, 1, n, 2)
            end

            p, q
        end

        function Base.getindex(H::$HierarchicalMatrix, i::Int, j::Int)
            p, q = size(H)
            M, N = blocksize(H)

            m = 1
            while m ≤ M
                r = blockgetsize(H, m, N, 1)
                if i > r
                    i -= r
                    m += 1
                else
                    break
                end
            end

            n = 1
            while n ≤ N
                s = blockgetsize(H, 1, n, 2)
                if j > s
                    j -= s
                    n += 1
                else
                    break
                end
            end

            blockgetindex(H, m, n, i, j)
        end

    end)
end
