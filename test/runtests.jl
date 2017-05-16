using HierarchicalMatrices
using Base.Test

for r in map(BLOCKRANK, subtypes(AbstractFloat))
    @test iseven(r)
end

srand(0)

for T in (Float32, Float64)
    A = rand(T, 10, 5)
    x = rand(T, 40)

    y = zero(x)
    HierarchicalMatrices.A_mul_B!(y, A, x, 1, 1)
    @test y[1:10] == A*x[1:5]

    fill!(y, 0.0)
    HierarchicalMatrices.A_mul_B!(y, A, x, 5, 5, 2, 2)
    @test y[5:2:23] == A*x[5:2:14]

    fill!(y, 0.0)
    HierarchicalMatrices.At_mul_B!(y, A, x, 1, 5, 2, 1)
    @test y[1:5] == A'x[5:2:23]

    fill!(y, 0.0)
    HierarchicalMatrices.At_mul_B!(y, A, x, 6, 3, 1, 3)
    @test y[6:3:18] == A'*x[3:12]

    A = map(big, A)
    x = map(big, x)

    y = zero(x)
    HierarchicalMatrices.A_mul_B!(y, A, x, 1, 1)
    @test y[1:10] == A*x[1:5]

    fill!(y, 0.0)
    HierarchicalMatrices.A_mul_B!(y, A, x, 5, 5, 2, 2)
    @test y[5:2:23] == A*x[5:2:14]

    fill!(y, 0.0)
    HierarchicalMatrices.At_mul_B!(y, A, x, 1, 5, 2, 1)
    @test y[1:5] == A'x[5:2:23]

    fill!(y, 0.0)
    HierarchicalMatrices.At_mul_B!(y, A, x, 6, 3, 1, 3)
    @test y[6:3:18] == A'*x[3:12]
end

include("../examples/Cauchy.jl")
