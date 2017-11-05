abstract type AbstractBarycentricMatrix{T} <: AbstractLowRankMatrix{T} end

size(B::AbstractBarycentricMatrix) = (B.b-B.a+1, B.d-B.c+1)

struct EvenBarycentricMatrix{T} <: AbstractBarycentricMatrix{T}
    a::Int
    b::Int
    c::Int
    d::Int
    x::Vector{T}
    λ::Vector{T}
    w::Vector{T}
    β::Vector{T}
    W::Matrix{T}
    F::Matrix{T}
end

function EvenBarycentricMatrix(::Type{T}, f::Function, a::Int, b::Int, c::Int, d::Int) where T
    n = BLOCKRANK(T)
    x = chebyshevpoints(T, n)
    λ = chebyshevbarycentricweights(T, n)

    w = zeros(T, b-a+1)
    β = zeros(T, n)
    @inbounds for i = a:b
        for k = 1:n
            w[i+1-a] += λ[k]*inv(2i-a-b-(b-a)*x[k])
        end
    end

    W = zeros(T, n, b-a+1)
    @inbounds for i = a:b
        for k = 1:n
            W[k, i-a+1] = λ[k]*inv((2i-a-b-(b-a)*x[k])*w[i+1-a])
        end
    end

    F = zeros(T, d-c+1, n)
    @inbounds for k = 1:n
        for j = c:d
            F[j-c+1, k] = f(T,(a+b)/2+(b-a)*x[k]/2,j)
        end
    end

    EvenBarycentricMatrix(a, b, c, d, x, λ, w, β, W, F)
end

function getindex(B::EvenBarycentricMatrix{T}, i::Int, j::Int) where T
    ret = zero(T)

    if iseven(size(B, 1)+size(B, 2)+i+j)
        @inbounds for k = 1:length(B.x)
            ret += B.F[j, k]*B.W[k, i]
        end
    end

    ret
end


function barycentricmatrix(::Type{T}, f::Function, a::Int, b::Int, c::Int, d::Int) where T
    n = BLOCKRANK(T)
    x = chebyshevpoints(T, n)
    λ = chebyshevbarycentricweights(T, n)

    w = zeros(T, b-a+1)
    β = zeros(T, n)
    @inbounds for i = a:b
        for k = 1:n
            w[i+1-a] += λ[k]*inv(2i-a-b-(b-a)*x[k])
        end
    end

    W = zeros(T, b-a+1, n)
    @inbounds for k = 1:n
        for i = a:b
            W[i-a+1, k] = λ[k]*inv((2i-a-b-(b-a)*x[k])*w[i+1-a])
        end
    end

    F = zeros(T, d-c+1, n)
    @inbounds for k = 1:n
        for j = c:d
            F[j-c+1, k] = f(T,(a+b)/2+(b-a)*x[k]/2,j)
        end
    end

    LowRankMatrix(W, Diagonal(ones(T, n)), F)
end


function chebyshevpoints(::Type{T}, n::Int; kind::Int = 1) where T
    x = zeros(T, n)
    nd2 = n÷2
    if kind == 1
        @inbounds for k = 1:nd2
            x[k] = sinpi((n-2k+one(T))/2n)
        end
        @inbounds for k=1:nd2
            x[n+1-k] = -x[k]
        end
    else
        @inbounds for k = 1:nd2
            x[k] = sinpi((n-2k+one(T))/(2*(n-1)))
        end
        @inbounds for k=1:nd2
            x[n+1-k] = -x[k]
        end
    end
    x
end


function chebyshevbarycentricweights(::Type{T}, n::Int; kind::Int = 1) where T
    λ = zeros(T, n)
    nd2 = n÷2
    if kind == 1
        @inbounds for k = 1:nd2+1
            λ[k] = sinpi((2k-one(T))/2n)
        end
        @inbounds for k=1:nd2
            λ[n+1-k] = λ[k]
        end
        @inbounds for k=2:2:n
            λ[k] *= -1
        end
    else
        fill!(λ, one(T))
        @inbounds for k=2:2:n
            λ[k] *= -1
        end
        λ[1] *= half(T)
        λ[n] *= half(T)
    end
    λ
end


struct BarycentricPoly2D{T}
    x::Vector{T}
    y::Vector{T}
    λx::Vector{T}
    λy::Vector{T}
    F::Matrix{T}
end

function BarycentricPoly2D(::Type{T}, f::Function, a::T, b::T, c::T, d::T) where T
    r = BLOCKRANK(T)
    rx = r
    ry = r
    x = chebyshevpoints(T, rx)
    y = chebyshevpoints(T, ry)
    λx = chebyshevbarycentricweights(T, rx)
    λy = chebyshevbarycentricweights(T, ry)

    ab2 = half(T)*(a+b)
    ba2 = half(T)*(b-a)
    @inbounds for p in 1:rx
        x[p] = ab2+ba2*x[p]
    end

    cd2 = half(T)*(c+d)
    dc2 = half(T)*(d-c)
    @inbounds for q in 1:ry
        y[q] = cd2+dc2*y[q]
    end

    F = zeros(T, rx, ry)

    @inbounds for n in 1:ry
        yn = y[n]
        for m in 1:rx
            F[m,n] = f(x[m], yn)
        end
    end

    BarycentricPoly2D(x, y, λx, λy, F)
end

function evaluate(B::BarycentricPoly2D{T}, x::T, y::T) where T
    xr, yr, λx, λy, F = B.x, B.y, B.λx, B.λy, B.F
    r = length(xr)
    temp1, temp2, temp3 = zero(T), zero(T), zero(T)
    ret = zero(T)
    for p in 1:r
        temp1 += λx[p]*inv(x-xr[p])
    end
    for q in 1:r
        temp2 += λy[q]*inv(y-yr[q])
    end

    for n in 1:r
        temp3 = zero(T)
        for m in 1:r
            temp3 += λx[m]*F[m,n]*inv(x-xr[m])
        end
        temp3 *= inv(temp1)
        ret += λy[n]*temp3*inv(y-yr[n])
    end
    ret *= inv(temp2)

    ret
end

(B::BarycentricPoly2D)(x, y) = evaluate(B, x, y)


struct BarycentricMatrix2D{T} <: AbstractBarycentricMatrix{T}
    B::BarycentricPoly2D{T}
    x::Vector{T}
    y::Vector{T}
    ir::UnitRange{Int}
    jr::UnitRange{Int}
    U::Matrix{T}
    V::Matrix{T}
    temp1::Vector{T}
    temp2::Vector{T}
end

size(B::BarycentricMatrix2D) = (length(B.ir), length(B.jr))

function getindex(B::BarycentricMatrix2D{T}, i::Int, j::Int) where T
    r = size(B.B.F, 1)
    ret = zero(T)
    for k in 1:r
        temp = zero(T)
        for l in 1:r
            temp += B.B.F[k, l]*B.V[j, l]
        end
        ret += B.U[i,k]*temp
    end

    ret
end

function BarycentricMatrix2D(::Type{T}, f::Function, a::T, b::T, c::T, d::T, x::Vector{T}, y::Vector{T}, ir::UnitRange{Int}, jr::UnitRange{Int}) where T
    BarycentricMatrix2D(BarycentricPoly2D(T, f, a, b, c, d), x, y, ir, jr)
end

function BarycentricMatrix2D(B::BarycentricPoly2D{T}, x::Vector{T}, y::Vector{T}, ir::UnitRange{Int}, jr::UnitRange{Int}) where T
    r = length(B.x)
    U = zeros(T, length(ir), r)
    V = zeros(T, length(jr), r)

    update!(BarycentricMatrix2D(B, x, y, ir, jr, U, V, zeros(T, r), zeros(T, r)), x, y, ir, jr)
end

function update!(B::BarycentricMatrix2D{T}, x::Vector{T}, y::Vector{T}, ir::UnitRange{Int}, jr::UnitRange{Int}) where T
    @assert length(ir) == length(B.ir)
    @assert length(jr) == length(B.jr)
    r = length(B.B.x)
    U, V = B.U, B.V

    ishift = 1-first(ir)
    jshift = 1-first(jr)
    for m in 1:r
        λxm = B.B.λx[m]
        xm = B.B.x[m]
        for i in ir
            B.U[i+ishift, m] = λxm*inv(x[i]-xm)
        end
    end
    for i in ir
        tempU = zero(T)
        for m in 1:r
            tempU += U[i+ishift, m]
        end
        for m in 1:r
            U[i+ishift, m] /= tempU
        end
    end
    for n in 1:r
        λyn = B.B.λy[n]
        yn = B.B.y[n]
        for j in jr
            V[j+jshift, n] = λyn*inv(y[j]-yn)
        end
    end
    for j in jr
        tempV = zero(T)
        for n in 1:r
            tempV += V[j+jshift, n]
        end
        for n in 1:r
            V[j+jshift, n] /= tempV
        end
    end

    for i in ir
        B.x[i] = x[i]
    end
    for j in jr
        B.y[j] = y[j]
    end

    B
end

function indsplit(x::Vector{T}, ir::UnitRange{Int}, a::T, b::T) where T
    i = first(ir)
    ab2 = half(T)*(a+b)
    while x[i] ≥ ab2
        i += 1
        i > last(ir) && break
    end
    first(ir):i-1, i:last(ir)
end
