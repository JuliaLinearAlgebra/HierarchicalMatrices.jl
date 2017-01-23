doc"""
`Block` is used for get index of a block matrix.  For example,
```julia
A[Block(1),Block(2)]
```
retrieves the 1 x 2 block
"""
immutable Block
    K::Int
end

Base.convert{T<:Integer}(::Type{T}, B::Block) = convert(T, B.K)::T

for OP in (:(Base.one), :(Base.zero), :(+), :(-))
    @eval $OP(B::Block) = Block($OP(B.K))
end

for OP in (:(==), :(!=), :(Base.isless))
    @eval $OP(A::Block,B::Block) = $OP(A.K,B.K)
end


for OP in (:(Base.rem), :(Base.div))
    @eval begin
        $OP(A::Block,B::Block) = Block($OP(A.K,B.K))
        $OP(A::Integer,B::Block) = Block($OP(A,B.K))
        $OP(A::Block,B::Integer) = Block($OP(A.K,B))
    end
end
