abstract AbstractBarycentricMatrix{T} <: AbstractLowRankMatrix{T}

size(B::AbstractBarycentricMatrix) = (B.b-B.a+1, B.d-B.c+1)

for BMAT in (:BarycentricMatrix,:EvenBarycentricMatrix)
    @eval begin

        immutable $BMAT{T} <: AbstractBarycentricMatrix{T}
            a::Int64
            b::Int64
            c::Int64
            d::Int64
            x::Vector{T}
            λ::Vector{T}
            w::Vector{T}
            β::Vector{T}
            W::Matrix{T}
            F::Matrix{T}
        end

        function $BMAT{T}(::Type{T}, f::Function, a::Int64, b::Int64, c::Int64, d::Int64)
            n = 18
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
                    W[k,i-a+1] = λ[k]*inv((2i-a-b-(b-a)*x[k])*w[i+1-a])
                end
            end

            F = zeros(T, d-c+1, n)
            @inbounds for k = 1:n
                for j = c:d
                    F[j-c+1,k] = f(T,(a+b)/2+(b-a)*x[k]/2,j)
                end
            end

            $BMAT(a,b,c,d,x,λ,w,β,W,F)
        end

    end
end

function getindex{T}(B::BarycentricMatrix{T}, i::Int, j::Int)
    ret = zero(T)

    @inbounds for k = 1:length(B.x)
        ret += B.F[j,k]*B.W[k,i]
    end

    ret
end

function getindex{T}(B::EvenBarycentricMatrix{T}, i::Int, j::Int)
    ret = zero(T)

    if iseven(B.a+B.c+i+j)
        @inbounds for k = 1:length(B.x)
            ret += B.F[j,k]*B.W[k,i]
        end
    end

    ret
end

function chebyshevpoints{T<:Number}(::Type{T}, n::Integer)
    x = zeros(T, n)
    nd2 = n÷2
    @inbounds for k = 1:nd2
        x[k] = sinpi((n-2k+one(T))/2n)
    end
    @inbounds for k=1:nd2
        x[n+1-k] = -x[k]
    end
    x
end

function chebyshevbarycentricweights{T<:Number}(::Type{T}, n::Integer)
    λ = zeros(T, n)
    nd2 = n÷2
    @inbounds for k = 1:nd2+1
        λ[k] = sinpi((2k-one(T))/2n)
    end
    @inbounds for k=1:nd2
        λ[n+1-k] = λ[k]
    end
    @inbounds for k=2:2:n
        λ[k] *= -1
    end
    λ
end
