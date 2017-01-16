__precompile__()
module HierarchicalMatrices
    const BLOCKSIZE = 200

    import Base: size, rank, norm, cond, istriu, istril, issymmetric, ishermitian, convert, view
    import Base: copy, getindex, setindex!, show, transpose, ctranspose, one, zero, inv, A_mul_B!
    import Base: Matrix, promote_op
    import Base: +, -, *, /, \, .+, .-, .*, ./, .\, ==
    import Base.LinAlg: checksquare, SingularException, arithtype
    import Base.BLAS: @blasfunc, libblas, BlasInt

    export BLOCKSIZE
    export AbstractHierarchicalMatrix, AbstractLowRankMatrix, AbstractBarycentricMatrix
    export LowRankMatrix, BarycentricMatrix, EvenBarycentricMatrix
    export @hierarchicalmatrix, blocksize, blockgetsize, blockgetindex, setblock!

    include("LowRankMatrix.jl")
    include("BarycentricMatrix.jl")
    include("HierarchicalMatrix.jl")
    include("algebra.jl")

end # module
