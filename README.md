# HierarchicalMatrices

This package provides a flexible framework for hierarchical matrices in Julia.

Create your own hierarchical matrix as simply as:
```julia
julia> using HierarchicalMatrices

julia> @hierarchicalmatrix HierarchicalMatrix LowRankMatrix Matrix

```
The invocation of the `@hierarchicalmatrix` macro creates the immutable type
`HierarchicalMatrix` and endows it with fields of `HierarchicalMatrixblocks`,
`LowRankMatrixblocks`, `Matrixblocks`, and a matrix of integers, `assigned`, to
determine which type of block is active.

See the example on speeding up the matrix-vector product with Cauchy matrices.

# Implementation

A straightforward implementation of hierarchical (self-referential) data types
might suffer from Russell's paradox. In the context of types, Russell's paradox
states that either you know the type, or you know its fields, but neither
concretely. On one side of the paradox, you end up with type-stable constructors
and type-unstable getters; on the other side, you are stuck with type-unstable
constructors and type-stable getters.

This implementation of hierarchical data types avoids Russell's paradox at the
cost of restricting the entire list of concrete matrix types that are the fields
upon construction. This allows for fast and type-stable setters and getters.
Enjoy!
