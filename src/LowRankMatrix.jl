abstract AbstractLowRankMatrix{T} <: AbstractMatrix{T}

"""
Store the singular value decomposition of a matrix:

    A = UΣV'

"""
immutable LowRankMatrix{T} <: AbstractLowRankMatrix{T}
    U::Matrix{T}
    Σ::Diagonal{T}
    V::Matrix{T}
    temp::Vector{T}
end

LowRankMatrix{T}(U::Matrix{T}, Σ::Diagonal{T}, V::Matrix{T}) = LowRankMatrix(U, Σ, V, zero(Σ.diag))

size(L::LowRankMatrix) = size(L.U, 1), size(L.V, 1)
rank(L::LowRankMatrix) = length(L.Σ.diag)
norm(L::LowRankMatrix) = first(L.Σ)
cond(L::LowRankMatrix) = ifelse(rank(L) < minimum(size(L)), Inf, first(L.Σ)/last(L.Σ))

istriu(L::LowRankMatrix) = false
istril(L::LowRankMatrix) = false
issymmetric(L::LowRankMatrix) = false
ishermitian(L::LowRankMatrix) = false

function getindex{T}(L::LowRankMatrix{T},i::Integer,j::Integer)
    ret = zero(T)
    U, Σ, V, r = L.U, L.Σ, L.V, rank(L)
    for k = r:-1:1
        ret += U[i,k]*Σ[k,k]*V[j,k]
    end

    ret
end

function convert{T}(::Type{LowRankMatrix{T}},L::LowRankMatrix)
    LowRankMatrix(convert(Matrix{T}, L.U), convert(Diagonal{T}, L.Σ), convert(Matrix{T}, L.V))
end

convert{T}(::Type{LowRankMatrix},A::AbstractMatrix{T}) = svdtrunc(A)

function getrank{T<:Real}(σ::Vector{T})
    r = length(σ)
    tol = r*eps(first(σ))
    while r ≥ 1
        if σ[r] > tol return r end
        r -= 1
    end

    r
end

function svdtrunc(A::AbstractMatrix)
    SVD = svdfact(A)
    r = getrank(SVD[:S])
    LowRankMatrix(SVD[:U][:,1:r], Diagonal(SVD[:S][1:r]), SVD[:V][:,1:r])
end

function (+){T}(L1::LowRankMatrix{T}, L2::LowRankMatrix{T})
    QRU = qrfact!(hcat(L1.U, L2.U))
    QRV = qrfact!(hcat(L1.V, L2.V))
    SVD = svdfact!(QRU[:R]*Diagonal(vcat(L1.Σ.diag, L2.Σ.diag))*QRV[:R]')
    r = getrank(SVD[:S])

    LowRankMatrix((QRU[:Q]*SVD[:U])[:,1:r], Diagonal(SVD[:S][1:r]), (QRV[:Q]*SVD[:V])[:,1:r])
end

function (-){T}(L1::LowRankMatrix{T}, L2::LowRankMatrix{T})
    QRU = qrfact!(hcat(L1.U, L2.U))
    QRV = qrfact!(hcat(L1.V, L2.V))
    SVD = svdfact!(QRU[:R]*Diagonal(vcat(L1.Σ.diag, -L2.Σ.diag))*QRV[:R]')
    r = getrank(SVD[:S])

    LowRankMatrix((QRU[:Q]*SVD[:U])[:,1:r], Diagonal(SVD[:S][1:r]), (QRV[:Q]*SVD[:V])[:,1:r])
end

function (*){T}(L1::LowRankMatrix{T}, L2::LowRankMatrix{T})
    SVD = svdfact!(L1.Σ*L1.V'*L2.U*L2.Σ)
    r = getrank(SVD[:S])

    LowRankMatrix((L1.U*SVD[:U])[:,1:r], Diagonal(SVD[:S][1:r]), (L2.V*SVD[:V])[:,1:r])
end

function (*){T}(A::AbstractMatrix{T}, L::LowRankMatrix{T})
    QRU = qrfact!(A*L.U)
    SVD = svdfact!(QRU[:R]*L.Σ)
    r = getrank(SVD[:S])

    LowRankMatrix((QRU[:Q]*SVD[:U])[:,1:r], Diagonal(SVD[:S][1:r]), (L.V*SVD[:V])[:,1:r])
end

function (*){T}(L::LowRankMatrix{T}, A::AbstractMatrix{T})
    QRV = qrfact!(A'*L.V)
    SVD = svdfact!(L.Σ*QRV[:R]')
    r = getrank(SVD[:S])

    LowRankMatrix((L.U*SVD[:U])[:,1:r], Diagonal(SVD[:S][1:r]), (QRV[:Q]*SVD[:V])[:,1:r])
end

function (*){T}(A::Diagonal{T}, L::LowRankMatrix{T})
    QRU = qrfact!(A*L.U)
    SVD = svdfact!(QRU[:R]*L.Σ)
    r = getrank(SVD[:S])

    LowRankMatrix((QRU[:Q]*SVD[:U])[:,1:r], Diagonal(SVD[:S][1:r]), (L.V*SVD[:V])[:,1:r])
end

function (*){T}(L::LowRankMatrix{T}, A::Diagonal{T})
    QRV = qrfact!(A'*L.V)
    SVD = svdfact!(L.Σ*QRV[:R]')
    r = getrank(SVD[:S])

    LowRankMatrix((L.U*SVD[:U])[:,1:r], Diagonal(SVD[:S][1:r]), (QRV[:Q]*SVD[:V])[:,1:r])
end

(*){T<:Number}(B::T, L::LowRankMatrix{T}) = LowRankMatrix(L.U, B*L.Σ, L.V)
(*){T<:Number}(L::LowRankMatrix{T}, B::T) = LowRankMatrix(L.U, L.Σ*B, L.V)
(/){T<:Number}(L::LowRankMatrix{T}, B::T) = LowRankMatrix(L.U, L.Σ/B, L.V)
(.*){T<:Number}(B::T, L::LowRankMatrix{T}) = LowRankMatrix(L.U, B.*L.Σ, L.V)
(.*){T<:Number}(L::LowRankMatrix{T}, B::T) = LowRankMatrix(L.U, L.Σ.*B, L.V)
(./){T<:Number}(L::LowRankMatrix{T}, B::T) = LowRankMatrix(L.U, L.Σ./B, L.V)
