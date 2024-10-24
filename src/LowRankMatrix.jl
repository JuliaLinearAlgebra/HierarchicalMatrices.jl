struct ThreadSafeVector{T} <: AbstractVector{T}
    V::Matrix{T}
    function ThreadSafeVector{T}(V::Matrix{T}) where T
        @assert size(V, 2) == Threads.nthreads()
        new{T}(V)
    end
end

ThreadSafeVector(V::Matrix) = ThreadSafeVector{eltype(V)}(V)
ThreadSafeVector(v::Vector) = ThreadSafeVector(repmat(v, 1, Threads.nthreads()))

@inline size(v::ThreadSafeVector) = (size(v.V, 1), )
@inline getindex(v::ThreadSafeVector{T}, i::Integer) where T = v.V[i, Threads.threadid()]
@inline setindex!(v::ThreadSafeVector{T}, x, i::Integer) where T = v.V[i, Threads.threadid()] = x

threadsafezeros(::Type{T}, n::Integer) where T = ThreadSafeVector{T}(zeros(T, n, Threads.nthreads()))
threadsafeones(::Type{T}, n::Integer) where T = ThreadSafeVector{T}(ones(T, n, Threads.nthreads()))

abstract type AbstractLowRankMatrix{T} <: AbstractMatrix{T} end


"""
Store the singular value decomposition of a matrix:

    A = UΣV'

"""
struct LowRankMatrix{T} <: AbstractLowRankMatrix{T}
    U::Matrix{T}
    Σ::Diagonal{T, Vector{T}}
    V::Matrix{T}
    temp::ThreadSafeVector{T}
end

LowRankMatrix(U::Matrix{T}, Σ::Diagonal{T}, V::Matrix{T}) where T = LowRankMatrix(U, Σ, V, threadsafezeros(T, length(Σ.diag)))
LowRankMatrix(U::Matrix{T}, Σ::Diagonal{T}, V::AbstractMatrix{T}) where T = LowRankMatrix(U, Σ, Matrix(V))
LowRankMatrix(U::AbstractMatrix{T}, Σ::Diagonal{T}, V::Matrix{T}) where T = LowRankMatrix(Matrix(U), Σ, V)
LowRankMatrix(U::AbstractMatrix{T}, Σ::Diagonal{T}, V::AbstractMatrix{T}) where T = LowRankMatrix(Matrix(U), Σ, Matrix(V))

size(L::LowRankMatrix) = size(L.U, 1), size(L.V, 1)
rank(L::LowRankMatrix) = length(L.Σ.diag)
norm(L::LowRankMatrix) = first(L.Σ)
cond(L::LowRankMatrix) = ifelse(rank(L) < minimum(size(L)), Inf, first(L.Σ)/last(L.Σ))

istriu(L::LowRankMatrix) = false
istril(L::LowRankMatrix) = false
issymmetric(L::LowRankMatrix) = false
ishermitian(L::LowRankMatrix) = false

function getindex(L::LowRankMatrix{T},i::Integer,j::Integer) where T
    ret = zero(T)
    U, Σ, V, r = L.U, L.Σ, L.V, rank(L)
    for k = r:-1:1
        ret += U[i,k]*Σ[k,k]*V[j,k]
    end

    ret
end

function getindex(L::LowRankMatrix, ir::UnitRange{Int}, jr::UnitRange{Int})
    LowRankMatrix(L.U[ir, :], L.Σ, L.V[jr, :])
end

function convert(::Type{LowRankMatrix{T}},L::LowRankMatrix) where T
    LowRankMatrix(convert(Matrix{T}, L.U), convert(Diagonal{T}, L.Σ), convert(Matrix{T}, L.V))
end

convert(::Type{LowRankMatrix},A::AbstractMatrix{T})  where T = svdtrunc(A)

function getrank(σ::Vector{T})  where T<:Real
    r = length(σ)
    r == 0 && (return 0)
    tol = r*eps(first(σ))
    while r ≥ 1
        if σ[r] > tol return r end
        r -= 1
    end

    r
end

function svdtrunc(A::AbstractMatrix)
    SVD = svd(A)
    r = getrank(SVD.S)
    LowRankMatrix(SVD.U[:,1:r], Diagonal(SVD.S[1:r]), SVD.V[:,1:r])
end

function lrzeros(::Type{T}, m::Int, n::Int) where T
    U = zeros(T, m, 0)
    Σ = Diagonal(zeros(T, 0))
    V = zeros(T, n, 0)
    LowRankMatrix(U, Σ, V)
end

function (+)(L1::LowRankMatrix{T}, L2::LowRankMatrix{T}) where T
    QRU = qr!(hcat(L1.U, L2.U))
    QRV = qr!(hcat(L1.V, L2.V))
    SVD = svd!(QRU.R*Diagonal(vcat(L1.Σ.diag, L2.Σ.diag))*QRV.R')
    r = getrank(SVD.S)

    LowRankMatrix((QRU.Q*SVD.U)[:,1:r], Diagonal(SVD.S[1:r]), (QRV.Q*SVD.V)[:,1:r])
end

function (-)(L1::LowRankMatrix{T}, L2::LowRankMatrix{T}) where T
    QRU = qr!(hcat(L1.U, L2.U))
    QRV = qr!(hcat(L1.V, L2.V))
    SVD = svd!(QRU.R*Diagonal(vcat(L1.Σ.diag, -L2.Σ.diag))*QRV.R')
    r = getrank(SVD.S)

    LowRankMatrix((QRU.Q*SVD.U)[:,1:r], Diagonal(SVD.S[1:r]), (QRV.Q*SVD.V)[:,1:r])
end

function (*)(L1::LowRankMatrix{T}, L2::LowRankMatrix{T}) where T
    SVD = svd!(L1.Σ*L1.V'L2.U*L2.Σ)
    r = getrank(SVD.S)

    LowRankMatrix((L1.U*SVD.U)[:,1:r], Diagonal(SVD.S[1:r]), (L2.V*SVD.V)[:,1:r])
end

function (*)(L1::Adjoint{T, <: LowRankMatrix{T}}, L2::LowRankMatrix{T}) where T
    L1P = parent(L1)
    SVD = svd!(L1P.Σ*L1P.U'L2.U*L2.Σ)
    r = getrank(SVD.S)

    LowRankMatrix((L1P.V*SVD.U)[:,1:r], Diagonal(SVD.S[1:r]), (L2.V*SVD.V)[:,1:r])
end

function (*)(A::AbstractMatrix{T}, L::LowRankMatrix{T}) where T
    QRU = qr!(A*L.U)
    SVD = svd!(QRU.R*L.Σ)
    r = getrank(SVD.S)

    LowRankMatrix((QRU.Q*SVD.U)[:,1:r], Diagonal(SVD.S[1:r]), (L.V*SVD.V)[:,1:r])
end

function (*)(L::LowRankMatrix{T}, A::AbstractMatrix{T}) where T
    QRV = qr!(A'L.V)
    SVD = svd!(L.Σ*QRV.R')
    r = getrank(SVD.S)

    LowRankMatrix((L.U*SVD.U)[:,1:r], Diagonal(SVD.S[1:r]), (QRV.Q*SVD.V)[:,1:r])
end

function (*)(A::Diagonal{T}, L::LowRankMatrix{T}) where T
    QRU = qr!(A*L.U)
    SVD = svd!(QRU.R*L.Σ)
    r = getrank(SVD.S)

    LowRankMatrix((QRU.Q*SVD.U)[:,1:r], Diagonal(SVD.S[1:r]), (L.V*SVD.V)[:,1:r])
end

function (*)(L::LowRankMatrix{T}, A::Diagonal{T}) where T
    QRV = qr!(A'L.V)
    SVD = svd!(L.Σ*QRV.R')
    r = getrank(SVD.S)

    LowRankMatrix((L.U*SVD.U)[:,1:r], Diagonal(SVD.S[1:r]), (QRV.Q*SVD.V)[:,1:r])
end

(*)(B::T, L::LowRankMatrix{T}) where {T<:Number} = LowRankMatrix(L.U, B*L.Σ, L.V)
(*)(L::LowRankMatrix{T}, B::T) where {T<:Number} = LowRankMatrix(L.U, L.Σ*B, L.V)
(/)(L::LowRankMatrix{T}, B::T) where {T<:Number} = LowRankMatrix(L.U, L.Σ/B, L.V)
function broadcast(::typeof(*), B::T, L::LowRankMatrix{T}) where T
    LowRankMatrix(L.U, B.*L.Σ, L.V)
end
function broadcast(::typeof(*), L::LowRankMatrix{T}, B::T) where T
    LowRankMatrix(L.U, L.Σ.*B, L.V)
end
function broadcast(::typeof(/), L::LowRankMatrix{T}, B::T) where T
    LowRankMatrix(L.U, L.Σ./B, L.V)
end
