macro hierarchical(HierarchicalType, Types...)
    blocks = :(begin end)
    push!(blocks.args, :($(Symbol("$(HierarchicalType)blocks"))::Matrix{$HierarchicalType{T}}))
    for i in 1:length(Types)
        push!(blocks.args, :($(Symbol("$(Types[i])blocks"))::Matrix{$(Types[i]){T}}))
    end
    AbstractHierarchicalType = parse("Abstract"*string(HierarchicalType))
    Factorization = parse(string(HierarchicalType)*"Factorization")
    return esc(quote
        import Base: +, -, *, /, \, .+, .-, .*, ./, .\, ==

        import HierarchicalMatrices: add_col!

        AbstractSuperType = promote_type(map(eval,$Types)...)

        export $AbstractHierarchicalType, $HierarchicalType, $Factorization

        abstract $AbstractHierarchicalType{T} <: AbstractSuperType{T}

        blocksize(H::$AbstractHierarchicalType) = size(H.assigned)

        function Base.size(H::$AbstractHierarchicalType)
            M, N = blocksize(H)

            p = 0
            for m = 1:M
                p += blocksize(H, m, N, 1)
            end

            q = 0
            for n = 1:N
                q += blocksize(H, 1, n, 2)
            end

            p, q
        end

        immutable $HierarchicalType{T} <: $AbstractHierarchicalType{T}
            $blocks
            assigned::Matrix{Int}
        end

        @generated function $HierarchicalType{T}(::Type{T}, M::Int, N::Int)
            L = length(fieldnames($HierarchicalType))
            HM = $HierarchicalType
            str = "$HM(Matrix{$HM}(M, N), "
            for l in 2:L-1
                S = $Types[l-1]
                str *= "Matrix{$S{T}}(M, N), "
            end
            str *= "zeros(Int, M, N))"
            return parse(str)
        end
        $HierarchicalType(M::Int, N::Int) = $HierarchicalType(Float64, M, N)

        immutable $Factorization{T} <: Factorization{T}
            $HierarchicalType::$HierarchicalType{T}
            factors::Matrix{Matrix{T}}
        end

        @generated function blocksize(H::$HierarchicalType, m::Int, n::Int)
            L = length(fieldnames(H))-1
            T = fieldname(H, 1)
            str = "
            begin
                Hmn = H.assigned[m,n]
                if Hmn == 1
                    return size(getindex(H.$T, m, n))"
            for l in 2:L
                T = fieldname(H, l)
                str *= "
                elseif Hmn == $l
                    return size(getindex(H.$T, m, n))"
            end
            str *= "
                end
                return (0,0)
            end"
            return parse(str)
        end

        @generated function blocksize(H::$HierarchicalType, m::Int, n::Int, k::Int)
            L = length(fieldnames(H))-1
            T = fieldname(H, 1)
            str = "
            begin
                Hmn = H.assigned[m,n]
                if Hmn == 1
                    return size(getindex(H.$T, m, n), k)"
            for l in 2:L
                T = fieldname(H, l)
                str *= "
                elseif Hmn == $l
                    return size(getindex(H.$T, m, n), k)"
            end
            str *= "
                end
                return 0
            end"
            return parse(str)
        end

        @generated function blockgetindex{S}(H::$HierarchicalType{S}, m::Int, n::Int, i::Int, j::Int)
            L = length(fieldnames(H))-1
            T = fieldname(H, 1)
            str = "
            begin
                Hmn = H.assigned[m,n]
                if Hmn == 1
                    return getindex(getindex(H.$T, m, n), i, j)"
            for l in 2:L
                T = fieldname(H, l)
                str *= "
                elseif Hmn == $l
                    return getindex(getindex(H.$T, m, n), i, j)"
            end
            str *= "
                end
                return zero(S)
            end"
            return parse(str)
        end

        @generated function Base.setindex!{S}(H::$HierarchicalType{S}, A::AbstractMatrix{S}, B1::Block, B2::Block)
            L = length(fieldnames(H))-1
            T = fieldname(H, 1)
            str = "
            begin
                m, n = B1.K, B2.K
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

        @generated function (+)(G::$HierarchicalType, H::$HierarchicalType)
            L = length(fieldnames(H))-1
            T = fieldname(H, 1)
            str = "
            begin
                F = deepcopy(G)
                M, N = blocksize(H)
                for m = 1:M
                    for n = 1:N
                        Gmn = G.assigned[m,n]
                        Hmn = H.assigned[m,n]
                        if Hmn == Gmn == 1
                            F[Block(m), Block(n)] = getindex(G.$T, m, n) + getindex(H.$T, m, n)"
            for l in 2:L
                T = fieldname(H, l)
                str *= "
                        elseif Hmn == Gmn == $l
                            F[Block(m), Block(n)] = getindex(G.$T, m, n) + getindex(H.$T, m, n)"
            end
            str *= "
                        end
                    end
                end
                return F
            end"
            return parse(str)
        end

        @generated function (-)(G::$HierarchicalType, H::$HierarchicalType)
            L = length(fieldnames(H))-1
            T = fieldname(H, 1)
            str = "
            begin
                F = deepcopy(G)
                M, N = blocksize(H)
                for m = 1:M
                    for n = 1:N
                        Gmn = G.assigned[m,n]
                        Hmn = H.assigned[m,n]
                        if Hmn == Gmn == 1
                            F[Block(m), Block(n)] = getindex(G.$T, m, n) - getindex(H.$T, m, n)"
            for l in 2:L
                T = fieldname(H, l)
                str *= "
                        elseif Hmn == Gmn == $l
                            F[Block(m), Block(n)] = getindex(G.$T, m, n) - getindex(H.$T, m, n)"
            end
            str *= "
                        end
                    end
                end
                return F
            end"
            return parse(str)
        end
    end)
end
