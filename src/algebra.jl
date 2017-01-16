for op in (:(+),:(-))
    @eval begin
        function $op{T}(B::AbstractBarycentricMatrix{T})
            B2 = deepcopy(B)
            B2.F[:] = $op(B.F)
            B2
        end
        function $op{T}(B1::BarycentricMatrix{T}, B2::BarycentricMatrix{T})
            @assert B1.a == B2.a && B1.b == B2.b && B1.c == B2.c && B1.d == B2.d
            @assert length(B1.x) == length(B2.x)
            B3 = deepcopy(B1)
            B3.F[:] = $op(B1.F, B2.F)
            B3
        end
        function $op{T}(B1::EvenBarycentricMatrix{T}, B2::EvenBarycentricMatrix{T})
            @assert B1.a == B2.a && B1.b == B2.b && B1.c == B2.c && B1.d == B2.d
            @assert length(B1.x) == length(B2.x)
            B3 = deepcopy(B1)
            B3.F[:] = $op(B1.F, B2.F)
            B3
        end
    end
end


# This file implements A_mul_B! overrides when A is a small matrix and u and v
# are larger vectors, but we want to operate in-place, equivalent to:
#
# u[istart:istart+size(A, 1)] += A*v[jstart:jstart+size(A, 2)]
#

# Generic Matrix
function A_mul_B!{T}(u::Vector{T}, A::AbstractMatrix{T}, v::AbstractVector{T}, istart::Int, jstart::Int)
    m, n = size(A)
    ishift, jshift = istart-1, jstart-1
    @inbounds for j = 1:n
        vj = v[jshift+j]
        for i = 1:m
            u[ishift+i] += A[i,j]*vj
        end
    end

    u
end

# BLAS'ed
for (fname, elty) in ((:dgemv_,:Float64),
                      (:sgemv_,:Float32),
                      (:zgemv_,:Complex128),
                      (:cgemv_,:Complex64))
    @eval begin
        function A_mul_B!(u::Vector{$elty}, A::Matrix{$elty}, v::Vector{$elty}, istart::Int, jstart::Int)
            ccall((@blasfunc($fname), libblas), Void,
                (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}),
                 &'N', &size(A,1), &size(A,2), &$elty(1.0),
                 A, &max(1,stride(A,2)), pointer(v, jstart), &1,
                 &$elty(1.0), pointer(u, istart), &1)
             u
        end
    end
end

# BarycentricMatrix

A_mul_B!{T}(u::Vector{T}, A::AbstractBarycentricMatrix{T}, v::AbstractVector{T}) = A_mul_B!(u, A, v, 1, 1)

function A_mul_B!{T}(u::Vector{T}, B::BarycentricMatrix{T}, v::AbstractVector{T}, istart::Int, jstart::Int)
    a, b, c, d = B.a, B.b, B.c, B.d
    x, λ, w, β, W, F = B.x, B.λ, B.w, B.β, B.W, B.F
    n = length(x)

    @inbounds for k = 1:n
        βk = zero(T)
        for j = c:d
            βk += v[jstart+j-c]*F[j-c+1,k]
        end
        β[k] = βk
    end

    @inbounds for i = a:b
        ui = zero(T)
        for k = 1:n
            ui += β[k]*W[k,i+1-a]
        end
        u[istart+i-a] += ui
    end

    u
end

function A_mul_B!{T}(u::Vector{T}, B::EvenBarycentricMatrix{T}, v::AbstractVector{T}, istart::Int, jstart::Int)
    a, b, c, d = B.a, B.b, B.c, B.d
    x, λ, w, β, W, F = B.x, B.λ, B.w, B.β, B.W, B.F
    n = length(x)

    if iseven(istart+jstart)
        @inbounds for k = 1:n
            βk = zero(T)
            for j = c:2:d
                βk += v[jstart+j-c]*F[j-c+1,k]
            end
            β[k] = βk
        end

        @inbounds for i = a:2:b
            ui = zero(T)
            for k = 1:n
                ui += β[k]*W[k,i+1-a]
            end
            u[istart+i-a] += ui
        end

        @inbounds for k = 1:n
            βk = zero(T)
            for j = c+1:2:d
                βk += v[jstart+j-c]*F[j-c+1,k]
            end
            β[k] = βk
        end

        @inbounds for i = a+1:2:b
            ui = zero(T)
            for k = 1:n
                ui += β[k]*W[k,i+1-a]
            end
            u[istart+i-a] += ui
        end
    else
        @inbounds for k = 1:n
            βk = zero(T)
            for j = c+1:2:d
                βk += v[jstart+j-c]*F[j-c+1,k]
            end
            β[k] = βk
        end

        @inbounds for i = a:2:b
            ui = zero(T)
            for k = 1:n
                ui += β[k]*W[k,i+1-a]
            end
            u[istart+i-a] += ui
        end

        @inbounds for k = 1:n
            βk = zero(T)
            for j = c:2:d
                βk += v[jstart+j-c]*F[j-c+1,k]
            end
            β[k] = βk
        end

        @inbounds for i = a+1:2:b
            ui = zero(T)
            for k = 1:n
                ui += β[k]*W[k,i+1-a]
            end
            u[istart+i-a] += ui
        end
    end

    u
end
