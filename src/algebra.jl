function unsafe_broadcasttimes!(a::Vector{T}, b::Vector{T}) where T
    @simd for i in 1:length(a)
        Base.unsafe_setindex!(a, Base.unsafe_getindex(a, i)*Base.unsafe_getindex(b, i), i)
    end
    a
end

for op in (:(+),:(-))
    @eval begin
        function $op(B::AbstractBarycentricMatrix{T}) where T
            B2 = deepcopy(B)
            B2.F[:] = $op(B.F)
            B2
        end
        function $op(B1::EvenBarycentricMatrix{T}, B2::EvenBarycentricMatrix{T}) where T
            @assert B1.a == B2.a && B1.b == B2.b && B1.c == B2.c && B1.d == B2.d
            @assert length(B1.x) == length(B2.x)
            B3 = deepcopy(B1)
            B3.F[:] = $op(B1.F, B2.F)
            B3
        end
    end
end


# This file implements mul! overrides when A is a small matrix and u and v
# are larger vectors, but we want to operate in-place, equivalent to:
#
# u[istart:istart+size(A, 1)-1] += A*v[jstart:jstart+size(A, 2)-1]
#

# Generic Matrix

mul!(y::AbstractVecOrMat{T}, A::AbstractMatrix{T}, x::AbstractVecOrMat{T}, istart::Int, jstart::Int)  where T =
    mul!(y, A, x, istart, jstart, 1, 1)

function mul!(y::AbstractVecOrMat{T}, A::AbstractMatrix{T}, x::AbstractVecOrMat{T}, istart::Int, jstart::Int, INCX::Int, INCY::Int) where T
    m, n = size(A)
    ishift, jshift = istart-INCY, jstart-INCX
    @inbounds for j = 1:n
        xj = x[jshift+j*INCX]
        for i = 1:m
            y[ishift+i*INCY] += A[i,j]*xj
        end
    end

    y
end

At_mul_B!(y::AbstractVecOrMat{T}, A::AbstractMatrix{T}, x::AbstractVecOrMat{T}, istart::Int, jstart::Int) where T = At_mul_B!(y, A, x, istart, jstart, 1, 1)

function At_mul_B!(y::AbstractVecOrMat{T}, A::AbstractMatrix{T}, x::AbstractVecOrMat{T}, istart::Int, jstart::Int, INCX::Int, INCY::Int) where T
    m, n = size(A)
    ishift, jshift = istart-INCY, jstart-INCX
    @inbounds for i = 1:n
        yi = zero(eltype(y))
        for j = 1:m
            yi += A[j,i]*x[jshift+j*INCX]
        end
        y[ishift+i*INCY] += yi
    end

    y
end

Ac_mul_B!(y::AbstractVecOrMat{T}, A::AbstractMatrix{T}, x::AbstractVecOrMat{T}, istart::Int, jstart::Int) where T = Ac_mul_B!(y, A, x, istart, jstart, 1, 1)

function Ac_mul_B!(y::AbstractVecOrMat{T}, A::AbstractMatrix{T}, x::AbstractVecOrMat{T}, istart::Int, jstart::Int, INCX::Int, INCY::Int) where T
    m, n = size(A)
    ishift, jshift = istart-INCY, jstart-INCX
    @inbounds for i = 1:n
        yi = zero(eltype(y))
        for j = 1:m
            yi += conj(A[j,i])*x[jshift+j*INCX]
        end
        y[ishift+i*INCY] += yi
    end

    y
end


# LowRankMatrix

mul!(y::AbstractVecOrMat{T}, L::LowRankMatrix{T}, x::AbstractVecOrMat{T}) where T = mul!(y, L, x, 1, 1)
mul!(y::AbstractVecOrMat{T}, L::LowRankMatrix{T}, x::AbstractVecOrMat{T}, istart::Int, jstart::Int) where T = mul!(y, L, x, istart, jstart, 1, 1)

function mul!(y::AbstractVecOrMat{T}, L::LowRankMatrix{T}, x::AbstractVecOrMat{T}, istart::Int, jstart::Int, INCX::Int, INCY::Int) where T
    m, n = size(L)
    ishift, jshift = istart-INCY, jstart-INCX
    temp = L.temp

    @inbounds for k = 1:rank(L)
        temp[k] = zero(T)
        for j = 1:n
            temp[k] += L.V[j,k]*x[jshift+j*INCX]
        end
        temp[k] *= L.Σ[k,k]
    end

    @inbounds for k = 1:rank(L)
        tempk = temp[k]
        for i = 1:m
            y[ishift+i*INCY] += L.U[i,k]*tempk
        end
    end

    y
end

# BLAS'ed
if VERSION < v"0.7-"
    #include("blas06.jl")
else
    #include("blas.jl")
end

# BarycentricMatrix

mul!(u::Vector{T}, A::AbstractBarycentricMatrix{T}, v::AbstractVector{T}) where T = mul!(u, A, v, 1, 1)

function mul!(u::Vector{T}, B::EvenBarycentricMatrix{T}, v::AbstractVector{T}, istart::Int, jstart::Int) where T
    β, W, F = B.β, B.W, B.F
    ishift, jshift, n = istart-1, jstart-1, length(β)

    if iseven(ishift+jshift)
        @inbounds for k = 1:n
            βk = zero(T)
            for j = 1:2:size(B, 2)
                βk += v[jshift+j]*F[j,k]
            end
            β[k] = βk
        end

        @inbounds for i = 1:2:size(B, 1)
            ui = zero(T)
            for k = 1:n
                ui += β[k]*W[k,i]
            end
            u[ishift+i] += ui
        end

        @inbounds for k = 1:n
            βk = zero(T)
            for j = 2:2:size(B, 2)
                βk += v[jshift+j]*F[j,k]
            end
            β[k] = βk
        end

        @inbounds for i = 2:2:size(B, 1)
            ui = zero(T)
            for k = 1:n
                ui += β[k]*W[k,i]
            end
            u[ishift+i] += ui
        end
    else
        @inbounds for k = 1:n
            βk = zero(T)
            for j = 2:2:size(B, 2)
                βk += v[jshift+j]*F[j,k]
            end
            β[k] = βk
        end

        @inbounds for i = 1:2:size(B, 1)
            ui = zero(T)
            for k = 1:n
                ui += β[k]*W[k,i]
            end
            u[ishift+i] += ui
        end

        @inbounds for k = 1:n
            βk = zero(T)
            for j = 1:2:size(B, 2)
                βk += v[jshift+j]*F[j,k]
            end
            β[k] = βk
        end

        @inbounds for i = 2:2:size(B, 1)
            ui = zero(T)
            for k = 1:n
                ui += β[k]*W[k,i]
            end
            u[ishift+i] += ui
        end
    end

    u
end

# BarycentricMatrix2D

function mul!(u::Vector{T}, B::BarycentricMatrix2D{T}, v::AbstractVector{T}, istart::Int, jstart::Int) where T
    U, F, V, temp1, temp2 = B.U, B.B.F, B.V, B.temp1, B.temp2
    ishift, jshift, r = istart-1, jstart-1, length(temp1)

    # temp1 = V'*v[jshift+j]

    for k in 1:r
        temp1k = zero(T)
        for j in 1:size(V, 1)
            temp1k += V[j,k]*v[jshift+j]
        end
        temp1[k] = temp1k
        temp2[k] = zero(T)
    end

    # temp2 = F*temp1

    for l in 1:r
        temp1l = temp1[l]
        for k in 1:r
            temp2[k] += F[k,l]*temp1l
        end
    end

    # u[ishift+i] = U*temp2

    for k in 1:r
        temp2k = temp2[k]
        for i in 1:size(U, 1)
            u[ishift+i] += U[i,k]*temp2k
        end
    end

    u
end


# C = A*Diagonal(b[jstart:jstart+size(A, 2)-1])

function scale!(C::Matrix, A::Matrix, b::Vector, jstart::Int)
    m, n = size(A)
    p, q = size(C)
    jshift = jstart-1
    @inbounds for j = 1:n
        bj = b[jshift+j]
        for i = 1:m
            C[i,j] = A[i,j]*bj
        end
    end
    C
end

function scale!(C::LowRankMatrix, A::LowRankMatrix, b::Vector, jstart::Int)
    scale!(C.V, b, A.V, jstart)
    C
end

# C = Diagonal(b[istart:istart+size(A, 1)-1])*A

function scale!(C::Matrix, b::Vector, A::Matrix, istart::Int)
    m, n = size(A)
    p, q = size(C)
    ishift = istart-1
    @inbounds for j = 1:n, i = 1:m
        C[i,j] = A[i,j]*b[ishift+i]
    end
    C
end

function scale!(C::LowRankMatrix, b::Vector, A::LowRankMatrix, istart::Int)
    scale!(C.U, b, A.U, istart)
    C
end


# Add e_j^T u[istart:istart+size(L, 1)-1] to A.

function add_col!(A::AbstractMatrix{T}, u::Vector{T}, istart::Int, j::Int) where T
    ishift, m = istart-1, size(A, 1)
    jm = (j-1)*m
    @simd for i = 1:m
        @inbounds A[i+jm] += u[i+ishift]
    end
    A
end

function add_col!(A::Matrix{T}, u::Vector{T}, istart::Int, j::Int) where T<:BlasReal
    m, n = size(A, 1), size(A, 2)
    BLAS.axpy!(m, one(T), pointer(u, istart), 1, pointer(A, (j-1)*m+1), 1)
    A
end

function add_col(A::AbstractMatrix{T}, u::Vector{T}, istart::Int, j::Int) where T
    B = deepcopy(A)
    ishift, m = istart-1, size(A, 1)
    jm = (j-1)*m
    @simd for i = 1:m
        @inbounds B[i+jm] += u[i+ishift]
    end
    B
end

function add_col(A::Matrix{T}, u::Vector{T}, istart::Int, j::Int) where T<:BlasReal
    m, n = size(A, 1), size(A, 2)
    B = Matrix{T}(undef, m, n)
    BLAS.blascopy!(m*n, A, 1, B, 1)
    BLAS.axpy!(m, one(T), pointer(u, istart), 1, pointer(B, (j-1)*m+1), 1)
    B
end

# Add e_j^T u[istart:istart+size(L, 1)-1] to UΣV^T.

function add_col(L::LowRankMatrix{T}, u::Vector{T}, istart::Int, j::Int) where T
    U, Σ, V = L.U, L.Σ, L.V
    U1 = hcat(U, u[istart:istart+size(L,1)-1])
    Σ1 = Diagonal(vcat(Σ.diag, one(T)))
    V1 = hcat(V, zeros(T, size(L, 2)))
    V1[j,rank(L)+1] = one(T)

    LowRankMatrix(U1,Σ1,V1)
end

function add_col(L::LowRankMatrix{T}, u::Vector{T}, istart::Int, j::Int) where T<:BlasReal
    U, Σ, V = L.U, L.Σ, L.V
    m, n, r, un = size(L, 1), size(L, 2), rank(L), one(T)
    U1 = Matrix{T}(undef, m, r+1)
    BLAS.blascopy!(m, pointer(u, istart), 1, U1, 1)
    BLAS.blascopy!(m*r, U, 1, pointer(U1, m+1), 1)
    diag = Vector{T}(r+1)
    BLAS.blascopy!(r, Σ.diag, 1, pointer(diag, 2), 1)
    Base.unsafe_setindex!(diag, un, 1)
    Σ1 = Diagonal(diag)
    V1 = zeros(T, n, r+1)
    BLAS.blascopy!(n*r, V, 1, pointer(V1, n+1), 1)
    Base.unsafe_setindex!(V1, un, j, 1)

    LowRankMatrix(U1,Σ1,V1)
end

function update!(::Type{T}, f::Function, A::AbstractMatrix{T}, x::Vector{T}, y::Vector{T}, ir::UnitRange{Int}, jr::UnitRange{Int}) where T
    ishift = 1-first(ir)
    jshift = 1-first(jr)

    for j in jr, i in ir
        A[i+ishift,j+jshift] = f(T, x[i], y[j])
    end

    A
end
