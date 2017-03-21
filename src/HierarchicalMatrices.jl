__precompile__()
module HierarchicalMatrices

    using Compat

    BLOCKSIZE(T) = round(Int, 4log(3+sqrt(T(8)), inv(eps(T))))
    BLOCKRANK(T) = round(Int, log(3+sqrt(T(8)), inv(eps(T))))

    import Base: size, rank, norm, cond, istriu, istril, issymmetric, ishermitian, convert, view
    import Base: copy, getindex, setindex!, show, transpose, ctranspose, one, zero, inv, A_mul_B!
    import Base: broadcast, scale!, Matrix, promote_op
    import Base: +, -, *, /, \, .+, .-, .*, ./, .\, ==, !=
    import Base.LinAlg: checksquare, SingularException, Factorization
    import Base.LinAlg: BlasInt, BlasFloat, BlasReal, BlasComplex
    import Base.BLAS: @blasfunc, libblas

    export BLOCKSIZE, BLOCKRANK, Block
    export AbstractLowRankMatrix, AbstractBarycentricMatrix
    export LowRankMatrix, BarycentricMatrix2D, EvenBarycentricMatrix
    export @hierarchical, barycentricmatrix, blocksize, blockgetindex
    export add_col, add_col!, lrzeros, indsplit

    """
    Compute a typed 0.5.
    """
    half(x::Number) = oftype(x,0.5)
    half(x::Integer) = half(float(x))
    half{T<:Number}(::Type{T}) = convert(T,0.5)
    half{T<:Integer}(::Type{T}) = half(AbstractFloat)

    """
    Compute a typed 2.
    """
    two(x::Number) = oftype(x,2)
    two{T<:Number}(::Type{T}) = convert(T,2)

    include("LowRankMatrix.jl")
    include("BarycentricMatrix.jl")
    include("block.jl")
    include("hierarchical.jl")
    include("HierarchicalMatrix.jl")
    include("algebra.jl")

end # module
