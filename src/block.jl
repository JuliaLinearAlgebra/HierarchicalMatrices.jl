"""
`Block` is used for get index of a block matrix.  For example,
```julia
A[Block(1),Block(2)]
```
retrieves the 1 x 2 block
"""
struct Block
    K::Int
end

convert(::Type{T}, B::Block) where T<:Integer = convert(T, B.K)::T

for OP in (:(one), :(zero), :(+), :(-))
    @eval $OP(B::Block) = Block($OP(B.K))
end

for OP in (:(==), :(!=), :(isless))
    @eval $OP(A::Block,B::Block) = $OP(A.K,B.K)
end


for OP in (:(div), :(rem))
    @eval begin
        $OP(A::Block,B::Block) = Block($OP(A.K,B.K))
        $OP(A::Integer,B::Block) = Block($OP(A,B.K))
        $OP(A::Block,B::Integer) = Block($OP(A.K,B))
    end
end
