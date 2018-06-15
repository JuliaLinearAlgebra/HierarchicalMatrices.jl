for (f!, char) in ((:mul!,  'N'),
                   (:At_mul_B!, 'T'),
                   (:Ac_mul_B!, 'C'))
    for (fname, elty) in ((:dgemv_, :Float64),
                          (:sgemv_, :Float32),
                          (:zgemv_, :Complex128),
                          (:cgemv_, :Complex64))
        @eval begin
            function $f!(y::VecOrMat{$elty}, A::Matrix{$elty}, x::VecOrMat{$elty}, istart::Int, jstart::Int, INCX::Int, INCY::Int)
                ccall((@blasfunc($fname), libblas), Void,
                    (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                     Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                     Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}),
                     &($char), &size(A,1), &size(A,2), &($elty(1.0)),
                     A, &max(1,stride(A,2)), pointer(x, jstart), &INCX,
                     &($elty(1.0)), pointer(y, istart), &INCY)
                 y
            end
        end
    end
end




for (fname, elty) in ((:dgemv_,:Float64),
                      (:sgemv_,:Float32),
                      (:zgemv_,:Complex128),
                      (:cgemv_,:Complex64))
    @eval begin
        function mul!(y::VecOrMat{$elty}, L::LowRankMatrix{$elty}, x::VecOrMat{$elty}, istart::Int, jstart::Int, INCX::Int, INCY::Int)
            fill!(L.temp, zero($elty))
            ccall((@blasfunc($fname), libblas), Void,
                (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}),
                 &'T', &size(L,2), &rank(L), &($elty(1.0)),
                 L.V, &max(1,stride(L.V,2)), pointer(x, jstart), &INCX,
                 &($elty(1.0)), L.temp, &1)
            unsafe_broadcasttimes!(L.temp, L.Σ.diag)
            ccall((@blasfunc($fname), libblas), Void,
                (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}),
                 &'N', &size(L,1), &rank(L), &$elty(1.0),
                 L.U, &max(1,stride(L.U,2)), L.temp, &1,
                 &$elty(1.0), pointer(y, istart), &INCY)

             y
        end
    end
end



for (fname, elty) in ((:dgemv_,:Float64),
                      (:sgemv_,:Float32),
                      (:zgemv_,:Complex128),
                      (:cgemv_,:Complex64))
    @eval begin
        function mul!(u::Vector{$elty}, B::BarycentricMatrix2D{$elty}, v::Vector{$elty}, istart::Int, jstart::Int)
            fill!(B.temp1, zero($elty))
            ccall((@blasfunc($fname), libblas), Void,
                (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}),
                 &'T', &size(B.V,1), &size(B.V,2), &$elty(1.0),
                 B.V, &max(1,stride(B.V,2)), pointer(v, jstart), &1,
                 &$elty(1.0), B.temp1, &1)
            fill!(B.temp2, zero($elty))
            ccall((@blasfunc($fname), libblas), Void,
                (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}),
                 &'N', &size(B.B.F,1), &size(B.B.F,2), &$elty(1.0),
                 B.B.F, &max(1,stride(B.B.F,2)), B.temp1, &1,
                 &$elty(1.0), B.temp2, &1)
            ccall((@blasfunc($fname), libblas), Void,
                (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}),
                 &'N', &size(B.U,1), &size(B.U,2), &$elty(1.0),
                 B.U, &max(1,stride(B.U,2)), B.temp2, &1,
                 &$elty(1.0), pointer(u, istart), &1)
             u
        end
    end
end
