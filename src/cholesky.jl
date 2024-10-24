#
# Assumes A is a symmetric hierarchical matrix, uses the upper storage blocks
# Dispatches on Val's of assigned blocks: a 2 x 2 SPD hierarchical matrix
#
# A = [A11 A12
#      A12' A22]
#
# cannot have low-rank blocks on the main diagonal. We also assume that A12 is
# either dense or low-rank. Thus, there are 8 possibilities.
#

function hierarchicalcholesky(A::HierarchicalMatrix)
    @assert blocksize(A) == (2, 2)
    M = A.assigned
    return hierarchicalcholesky(A, Val(M[1, 1]), Val(M[1, 2]), Val(M[2, 2]))
end

function hierarchicalcholesky(A::Matrix)
    Matrix(cholesky(Symmetric(A)).U)
end

function hierarchicalcholesky(A::HierarchicalMatrix{T}, ::Val{1}, ::Val{2}, ::Val{1}) where T
    R = HierarchicalMatrix(T, 2, 2)
    R[Block(1), Block(1)] = R11 = hierarchicalcholesky(A.HierarchicalMatrixblocks[1, 1])
    R[Block(1), Block(2)] = R12 = solvetransposed(R11, A.LowRankMatrixblocks[1, 2])
    A22mR12tR12 = A.HierarchicalMatrixblocks[2, 2] - R12'R12
    R[Block(2), Block(2)] = hierarchicalcholesky(A22mR12tR12)
    return R
end

function hierarchicalcholesky(A::HierarchicalMatrix{T}, ::Val{1}, ::Val{2}, ::Val{3}) where T
    R = HierarchicalMatrix(T, 2, 2)
    R[Block(1), Block(1)] = R11 = hierarchicalcholesky(A.HierarchicalMatrixblocks[1, 1])
    R[Block(1), Block(2)] = R12 = solvetransposed(R11, A.LowRankMatrixblocks[1, 2])
    A22mR12tR12 = A.Matrixblocks[2, 2] - R12'R12
    R[Block(2), Block(2)] = hierarchicalcholesky(A22mR12tR12)
    return R
end

function hierarchicalcholesky(A::HierarchicalMatrix{T}, ::Val{1}, ::Val{3}, ::Val{1}) where T
    R = HierarchicalMatrix(T, 2, 2)
    R[Block(1), Block(1)] = R11 = hierarchicalcholesky(A.HierarchicalMatrixblocks[1, 1])
    R[Block(1), Block(2)] = R12 = solvetransposed(R11, A.Matrixblocks[1, 2])
    A22mR12tR12 = A.HierarchicalMatrixblocks[2, 2] - R12'R12
    R[Block(2), Block(2)] = hierarchicalcholesky(A22mR12tR12)
    return R
end

function hierarchicalcholesky(A::HierarchicalMatrix{T}, ::Val{1}, ::Val{3}, ::Val{3}) where T
    R = HierarchicalMatrix(T, 2, 2)
    R[Block(1), Block(1)] = R11 = hierarchicalcholesky(A.HierarchicalMatrixblocks[1, 1])
    R[Block(1), Block(2)] = R12 = solvetransposed(R11, A.Matrixblocks[1, 2])
    A22mR12tR12 = A.Matrixblocks[2, 2] - R12'R12
    R[Block(2), Block(2)] = hierarchicalcholesky(A22mR12tR12)
    return R
end

function hierarchicalcholesky(A::HierarchicalMatrix{T}, ::Val{3}, ::Val{2}, ::Val{1}) where T
    R = HierarchicalMatrix(T, 2, 2)
    R[Block(1), Block(1)] = R11 = hierarchicalcholesky(A.Matrixblocks[1, 1])
    R[Block(1), Block(2)] = R12 = solvetransposed(R11, A.LowRankMatrixblocks[1, 2])
    A22mR12tR12 = A.HierarchicalMatrixblocks[2, 2] - R12'R12
    R[Block(2), Block(2)] = hierarchicalcholesky(A22mR12tR12)
    return R
end

function hierarchicalcholesky(A::HierarchicalMatrix{T}, ::Val{3}, ::Val{2}, ::Val{3}) where T
    R = HierarchicalMatrix(T, 2, 2)
    R[Block(1), Block(1)] = R11 = hierarchicalcholesky(A.Matrixblocks[1, 1])
    R[Block(1), Block(2)] = R12 = solvetransposed(R11, A.LowRankMatrixblocks[1, 2])
    A22mR12tR12 = A.Matrixblocks[2, 2] - R12'R12
    R[Block(2), Block(2)] = hierarchicalcholesky(A22mR12tR12)
    return R
end

function hierarchicalcholesky(A::HierarchicalMatrix{T}, ::Val{3}, ::Val{3}, ::Val{1}) where T
    R = HierarchicalMatrix(T, 2, 2)
    R[Block(1), Block(1)] = R11 = hierarchicalcholesky(A.Matrixblocks[1, 1])
    R[Block(1), Block(2)] = R12 = solvetransposed(R11, A.Matrixblocks[1, 2])
    A22mR12tR12 = A.HierarchicalMatrixblocks[2, 2] - R12'R12
    R[Block(2), Block(2)] = hierarchicalcholesky(A22mR12tR12)
    return R
end

function hierarchicalcholesky(A::HierarchicalMatrix{T}, ::Val{3}, ::Val{3}, ::Val{3}) where T
    R = HierarchicalMatrix(T, 2, 2)
    R[Block(1), Block(1)] = R11 = hierarchicalcholesky(A.Matrixblocks[1, 1])
    R[Block(1), Block(2)] = R12 = solvetransposed(R11, A.Matrixblocks[1, 2])
    A22mR12tR12 = A.Matrixblocks[2, 2] - R12'R12
    R[Block(2), Block(2)] = hierarchicalcholesky(A22mR12tR12)
    return R
end



# out = R⁻ᵀ L
function solvetransposed(R::HierarchicalMatrix{T}, L::LowRankMatrix{T}) where T
    U = L.U
    r = size(U, 2)
    Û = zero(U)
    for j in 1:r
        Û[:, j] = solvetransposed(R, U[:, j])
    end
    return HierarchicalMatrices.LowRankMatrix(Û, L.Σ, L.V)
end

function solvetransposed(R::HierarchicalMatrix{T}, M::Matrix{T}) where T
    r = size(M, 2)
    M̂ = zero(M)
    for j in 1:r
        M̂[:, j] = solvetransposed(R, M[:, j])
    end
    return M̂
end

function solvetransposed(R::Matrix{T}, L::LowRankMatrix{T}) where T
    U = L.U
    r = size(U, 2)
    Û = zero(U)
    for j in 1:r
        Û[:, j] = solvetransposed(R, U[:, j])
    end
    return HierarchicalMatrices.LowRankMatrix(Û, L.Σ, L.V)
end

function solvetransposed(R::Matrix{T}, M::Matrix{T}) where T
    r = size(M, 2)
    M̂ = zero(M)
    for j in 1:r
        M̂[:, j] = solvetransposed(R, M[:, j])
    end
    return M̂
end






function solvetransposed(R::Matrix{T}, b::Vector{T}) where T
    x = UpperTriangular(R)'\b
end

function solvetransposed(R::HierarchicalMatrix{T}, b::Vector{T}) where T
    @assert blocksize(R) == (2, 2)
    M = R.assigned
    return solvetransposed(R, b, Val(M[1, 1]), Val(M[1, 2]), Val(M[2, 2]))
end

function solvetransposed(R::HierarchicalMatrix{T}, b::Vector{T}, ::Val{1}, ::Val{2}, ::Val{1}) where T
    n = size(R, 2)
    s = size(R.HierarchicalMatrixblocks[1, 1], 2)
    b1 = b[1:s]
    b2 = b[s+1:n]
    x1 = solvetransposed(R.HierarchicalMatrixblocks[1, 1], b1)
    x2 = solvetransposed(R.HierarchicalMatrixblocks[2, 2], b2 - R.LowRankMatrixblocks[1, 2]'x1)
    return [x1; x2]
end

function solvetransposed(R::HierarchicalMatrix{T}, b::Vector{T}, ::Val{1}, ::Val{2}, ::Val{3}) where T
    n = size(R, 2)
    s = size(R.HierarchicalMatrixblocks[1, 1], 2)
    b1 = b[1:s]
    b2 = b[s+1:n]
    x1 = solvetransposed(R.HierarchicalMatrixblocks[1, 1], b1)
    x2 = solvetransposed(R.Matrixblocks[2, 2], b2 - R.LowRankMatrixblocks[1, 2]'x1)
    return [x1; x2]
end

function solvetransposed(R::HierarchicalMatrix{T}, b::Vector{T}, ::Val{1}, ::Val{3}, ::Val{1}) where T
    n = size(R, 2)
    s = size(R.HierarchicalMatrixblocks[1, 1], 2)
    b1 = b[1:s]
    b2 = b[s+1:n]
    x1 = solvetransposed(R.HierarchicalMatrixblocks[1, 1], b1)
    x2 = solvetransposed(R.HierarchicalMatrixblocks[2, 2], b2 - R.Matrixblocks[1, 2]'x1)
    return [x1; x2]
end

function solvetransposed(R::HierarchicalMatrix{T}, b::Vector{T}, ::Val{1}, ::Val{3}, ::Val{3}) where T
    n = size(R, 2)
    s = size(R.HierarchicalMatrixblocks[1, 1], 2)
    b1 = b[1:s]
    b2 = b[s+1:n]
    x1 = solvetransposed(R.HierarchicalMatrixblocks[1, 1], b1)
    x2 = solvetransposed(R.Matrixblocks[2, 2], b2 - R.Matrixblocks[1, 2]'x1)
    return [x1; x2]
end

function solvetransposed(R::HierarchicalMatrix{T}, b::Vector{T}, ::Val{3}, ::Val{2}, ::Val{1}) where T
    n = size(R, 2)
    s = size(R.Matrixblocks[1, 1], 2)
    b1 = b[1:s]
    b2 = b[s+1:n]
    x1 = solvetransposed(R.Matrixblocks[1, 1], b1)
    x2 = solvetransposed(R.HierarchicalMatrixblocks[2, 2], b2 - R.LowRankMatrixblocks[1, 2]'x1)
    return [x1; x2]
end

function solvetransposed(R::HierarchicalMatrix{T}, b::Vector{T}, ::Val{3}, ::Val{2}, ::Val{3}) where T
    n = size(R, 2)
    s = size(R.Matrixblocks[1, 1], 2)
    b1 = b[1:s]
    b2 = b[s+1:n]
    x1 = solvetransposed(R.Matrixblocks[1, 1], b1)
    x2 = solvetransposed(R.Matrixblocks[2, 2], b2 - R.LowRankMatrixblocks[1, 2]'x1)
    return [x1; x2]
end

function solvetransposed(R::HierarchicalMatrix{T}, b::Vector{T}, ::Val{3}, ::Val{3}, ::Val{1}) where T
    n = size(R, 2)
    s = size(R.Matrixblocks[1, 1], 2)
    b1 = b[1:s]
    b2 = b[s+1:n]
    x1 = solvetransposed(R.Matrixblocks[1, 1], b1)
    x2 = solvetransposed(R.HierarchicalMatrixblocks[2, 2], b2 - R.Matrixblocks[1, 2]'x1)
    return [x1; x2]
end

function solvetransposed(R::HierarchicalMatrix{T}, b::Vector{T}, ::Val{3}, ::Val{3}, ::Val{3}) where T
    n = size(R, 2)
    s = size(R.Matrixblocks[1, 1], 2)
    b1 = b[1:s]
    b2 = b[s+1:n]
    x1 = solvetransposed(R.Matrixblocks[1, 1], b1)
    x2 = solvetransposed(R.Matrixblocks[2, 2], b2 - R.Matrixblocks[1, 2]'x1)
    return [x1; x2]
end
