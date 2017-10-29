# In this example, we write a simple method to speed up the matrix-vector
# product of where the matrix is defined by sampling an asymptotically
# smooth kernel K(x-y):
#
# K(x_i-y_j), for 1 ≤ i,j ≤ N
#
# Examples include the Cauchy kernel:
#
# K(x-y) = 1/(x-y),
#
# the Coulomb kernel:
#
# K(x-y) = 1/(x-y)^2,
#
# a kernel proportional to its derivative:
#
# K(x-y) = 1/(x-y)^3,
#
# and the logarithmic kernel:
#
# K(x-y) = log|x-y|.
#
# Many kernels are asymptotically smooth, and a hierarchical decomposition
# with bounded off-diagonal numerical rank that satisfies the strong
# admissibility criterion is justified.
#

using HierarchicalMatrices

@inline cauchykernel{T}(x::T, y::T) = inv(x-y)
@inline coulombkernel{T}(x::T, y::T) = inv((x-y)^2)
@inline coulombprimekernel{T}(x::T, y::T) = inv((x-y)^3)
@inline logkernel{T}(x::T, y::T) = log(abs(x-y))

for f in (:cauchykernel, :coulombkernel, :coulombprimekernel, :logkernel)
    @eval $f{T}(x::Vector{T}, y::Vector{T}) = T[$f(x, y) for x in x, y in y]
end

#
# It is in the large-N asymptotic regime that the hierarchical approach
# demonstrates quasi-linear scaling, whereas normally we would expect quadratic
# scaling (beyond the BLAS'ed sub-sizes).
#
# We demonstrate with two different pairs of point distributions
# that are followed by both source and target points x and y:
#
#  - quadratic clustering near ±1 as in Chebyshev points.
#  - quadratic spacing on [0,∞) as in the spectra of second order linear
#    differential equations.
#

import HierarchicalMatrices: chebyshevpoints

for f in (cauchykernel, coulombkernel, coulombprimekernel, logkernel)
    for N in (1000,10_000)
        b = randn(N)
        x = chebyshevpoints(Float64, N)
        y = chebyshevpoints(Float64, N; kind = 2)
        println()
        println("$f matrix construction at N = $N")
        println()
        @time K = KernelMatrix(f, x, y, 1.0, -1.0, 1.0, -1.0)
        @time K = KernelMatrix(f, x, y, 1.0, -1.0, 1.0, -1.0)
        @time KF = f(x, y)
        @time KF = f(x, y)
        println()
        println("$f matrix-vector multiplication at N = $N")
        println()
        @time K*b
        @time K*b
        @time KF*b
        @time KF*b
        println()
        println("2-norm relative error: ",norm(K*b - KF*b)/norm(K*b))
        println()
    end
end

println()

for f in (cauchykernel, coulombkernel, coulombprimekernel, logkernel)
    for N in (1000, 10_000)
        b = randn(N)
        x = Float64[i*(i+1) for i = N:-1:1]
        y = Float64[(i+1/2)*(i+3/2) for i = N:-1:1]
        xmin, xmax = extrema(x)
        ymin, ymax = extrema(y)
        println()
        println("$f matrix construction at N = $N")
        println()
        @time K = KernelMatrix(f, x, y, xmax, xmin, ymax, ymin)
        @time K = KernelMatrix(f, x, y, xmax, xmin, ymax, ymin)
        @time KF = f(x, y)
        @time KF = f(x, y)
        println()
        println("$f matrix-vector multiplication at N = $N")
        println()
        @time K*b
        @time K*b
        @time KF*b
        @time KF*b
        println()
        println("2-norm relative error: ",norm(K*b - KF*b)/norm(K*b))
        println()
    end
end
