# HierarchicalMatrices.jl

[![Build Status](https://travis-ci.org/JuliaMatrices/HierarchicalMatrices.jl.svg?branch=master)](https://travis-ci.org/JuliaMatrices/HierarchicalMatrices.jl)  [![AppVeyor](https://ci.appveyor.com/api/projects/status/1t01s8cuoxrriem4/branch/master?svg=true)](https://ci.appveyor.com/project/MikaelSlevinsky/hierarchicalmatrices-jl-xfd1e/branch/master)

This package provides a flexible framework for hierarchical data types in Julia.

Create your own hierarchical matrix as simply as:
```julia
julia> using HierarchicalMatrices

julia> @hierarchical MyHierarchicalMatrix LowRankMatrix Matrix

```
The invocation of the `@hierarchical` macro creates an abstract supertype
`AbstractMyHierarchicalMatrix{T} <: AbstractMatrix{T}` and the immutable type
`MyHierarchicalMatrix`, endowing it with fields of `HierarchicalMatrixblocks`,
`LowRankMatrixblocks`, `Matrixblocks`, and a matrix of integers, `assigned`, to
determine which type of block is active. The package comes pre-loaded with a
`HierarchicalMatrix`.

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
