__precompile__()
module HierarchicalMatrices
    using Compat, Compat.LinearAlgebra

    BLOCKRANK(T::Type{R}) where {R<:Real} = 2round(Int, half(T)*log(3+sqrt(T(8)), inv(eps(T))))
    BLOCKRANK(T::Type{C}) where {C<:Complex} = BLOCKRANK(real(T))
    BLOCKSIZE(T) = 4BLOCKRANK(T)

    import Base: convert, view, size
    import Base: copy, getindex, setindex!, show, one, zero, inv, isless
    import Base: div, rem
    import Base: broadcast, scale!, Matrix, promote_op
    import Base: +, -, *, /, \, .+, .-, .*, ./, .\, ==, !=
    import Compat.LinearAlgebra: Factorization, rank, norm, cond, istriu, istril, issymmetric, ishermitian,
                                   transpose
    import Compat: adjoint                                 
    import Compat.LinearAlgebra.BLAS: @blasfunc, libblas, BlasInt, BlasFloat, BlasReal, BlasComplex

    export BLOCKSIZE, BLOCKRANK, Block
    export AbstractLowRankMatrix, AbstractBarycentricMatrix
    export LowRankMatrix, BarycentricMatrix2D, EvenBarycentricMatrix
    export @hierarchical, barycentricmatrix, blocksize, blockgetindex
    export add_col, add_col!, lrzeros, indsplit

    """
    Compute a typed 0.5.
    """
    half(x::Number) = oftype(x, 0.5)
    half(x::Integer) = half(float(x))
    half(::Type{T}) where {T<:Number} = convert(T, 0.5)
    half(::Type{T}) where {T<:Integer} = half(AbstractFloat)

    """
    Compute a typed 2.
    """
    two(x::Number) = oftype(x,2)
    two(::Type{T}) where {T<:Number} = convert(T, 2)

    for op in (:A_mul_B!, :At_mul_B!, :Ac_mul_B!, :scale!)
        @eval begin
            $op(args...) = Base.$op(args...)
        end
    end

    include("LowRankMatrix.jl")
    include("BarycentricMatrix.jl")
    include("block.jl")
    include("hierarchical.jl")
    include("HierarchicalMatrix.jl")
    include("KernelMatrix.jl")
    include("algebra.jl")

end # module
