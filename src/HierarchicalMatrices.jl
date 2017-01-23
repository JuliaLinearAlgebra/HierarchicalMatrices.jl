__precompile__()
module HierarchicalMatrices
    const BLOCKSIZE = 200

    import Base: size, rank, norm, cond, istriu, istril, issymmetric, ishermitian, convert, view
    import Base: copy, getindex, setindex!, show, transpose, ctranspose, one, zero, inv, A_mul_B!
    import Base: scale!, Matrix, promote_op
    import Base: +, -, *, /, \, .+, .-, .*, ./, .\, ==, !=
    import Base.LinAlg: checksquare, SingularException, arithtype, Factorization
    import Base.LinAlg: BlasInt, BlasFloat, BlasReal, BlasComplex
    import Base.BLAS: @blasfunc, libblas

    export BLOCKSIZE, Block
    export AbstractLowRankMatrix, AbstractBarycentricMatrix
    export LowRankMatrix, barycentricmatrix, EvenBarycentricMatrix
    export @hierarchical, blocksize, blockgetindex
    export add_col, add_col!, lrzeros

    include("LowRankMatrix.jl")
    include("BarycentricMatrix.jl")
    include("block.jl")
    include("hierarchical.jl")
    include("HierarchicalMatrix.jl")
    include("algebra.jl")

end # module
