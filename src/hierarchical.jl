function blocksize end
function blockgetindex end

macro hierarchical(HierarchicalType, Types...)
    blocks = :(begin end)
    push!(blocks.args, :($(Symbol("$(HierarchicalType)blocks"))::Matrix{$HierarchicalType{T}}))
    for i in 1:length(Types)
        if typeof(Types[i]) == Symbol
            push!(blocks.args, :($(Symbol("$(Types[i])blocks"))::Matrix{$(Types[i]){T}}))
        elseif typeof(Types[i]) == Expr
            push!(blocks.args, :($(Symbol("$(Types[i].args[1])blocks"))::Matrix{$(Types[i].args[1]){T,$(Types[i].args[2]){T}}}))
        else
            error("Invalid hierarchical symbol.")
        end
    end
    AbstractHierarchicalType = Meta.parse("Abstract"*string(HierarchicalType))
    Factorization = Meta.parse(string(HierarchicalType)*"Factorization")
    return esc(quote
        import Base: +, -, *, /, \, .+, .-, .*, ./, .\, ==
        import Base: size, getindex, setindex!
        import Compat: undef
        import Compat.LinearAlgebra: Factorization

        # import HierarchicalMatrices:

        AbstractSuperType = promote_type([typeof($Types[i]) == Symbol ? eval($Types[i]) : eval($Types[i].args[1]) for i in 1:length($Types)]...)

        export $AbstractHierarchicalType, $HierarchicalType, $Factorization

        abstract type $AbstractHierarchicalType{T} <: AbstractSuperType{T} end

        HierarchicalMatrices.blocksize(H::$AbstractHierarchicalType) = size(H.assigned)

        function size(H::$AbstractHierarchicalType)
            M, N = HierarchicalMatrices.blocksize(H)

            p = 0
            for m = 1:M
                p += HierarchicalMatrices.blocksize(H, m, N, 1)
            end

            q = 0
            for n = 1:N
                q += HierarchicalMatrices.blocksize(H, 1, n, 2)
            end

            p, q
        end

        struct $HierarchicalType{T} <: $AbstractHierarchicalType{T}
            $blocks
            assigned::Matrix{Int}
        end

        @generated function $HierarchicalType(::Type{T}, M::Int, N::Int) where T
            L = length(fieldnames($HierarchicalType))
            HM = $HierarchicalType
            str = VERSION < v"0.6-" ? "$HM(Matrix{$HM}(undef, M, N), " : "$HM(Matrix{$HM{T}}(undef, M, N), "
            for l in 2:L-1
                S = $Types[l-1]
                if typeof(S) == Symbol
                    str *= "Matrix{$S{T}}(undef, M, N), "
                else
                    str *= "Matrix{$(S.args[1]){T, $(S.args[2]){T}}}(undef, M, N), "
                end
            end
            str *= "zeros(Int, M, N))"
            return Meta.parse(str)
        end
        $HierarchicalType(M::Int, N::Int) = $HierarchicalType(Float64, M, N)

        struct $Factorization{T} <: Factorization{T}
            $HierarchicalType::$HierarchicalType{T}
            factors::Matrix{Matrix{T}}
        end

        HierarchicalMatrices.blocksize(H::$HierarchicalType, m::Int, n::Int) = HierarchicalMatrices.blocksize(H, m, n, 1), HierarchicalMatrices.blocksize(H, m, n, 2)

        @generated function HierarchicalMatrices.blocksize(H::$HierarchicalType, m::Int, n::Int, k::Int)
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
            return Meta.parse(str)
        end

        @generated function HierarchicalMatrices.blockgetindex(H::$HierarchicalType{S}, m::Int, n::Int, i::Int, j::Int) where S
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
            return Meta.parse(str)
        end

        function getindex(H::$HierarchicalType, i::Int, j::Int)
            p, q = size(H)
            M, N = HierarchicalMatrices.blocksize(H)

            m = 1
            while m ≤ M
                r = HierarchicalMatrices.blocksize(H, m, N, 1)
                if i > r
                    i -= r
                    m += 1
                else
                    break
                end
            end

            n = 1
            while n ≤ N
                s = HierarchicalMatrices.blocksize(H, 1, n, 2)
                if j > s
                    j -= s
                    n += 1
                else
                    break
                end
            end

            HierarchicalMatrices.blockgetindex(H, m, n, i, j)
        end

        @generated function setindex!(H::$HierarchicalType{S}, A::AbstractMatrix{S}, B1::Block, B2::Block) where S
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
            return Meta.parse(str)
        end

        @generated function (+)(G::$HierarchicalType, H::$HierarchicalType)
            L = length(fieldnames(H))-1
            T = fieldname(H, 1)
            str = "
            begin
                F = deepcopy(G)
                M, N = HierarchicalMatrices.blocksize(H)
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
            return Meta.parse(str)
        end

        @generated function (-)(G::$HierarchicalType, H::$HierarchicalType)
            L = length(fieldnames(H))-1
            T = fieldname(H, 1)
            str = "
            begin
                F = deepcopy(G)
                M, N = HierarchicalMatrices.blocksize(H)
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
            return Meta.parse(str)
        end
    end)
end
