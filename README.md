<img align="right" width="26%" src="./misc/logo.png">

# Optinum

SIMD-accelerated tensor math and numerical optimization - some assembly required

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER-FACING API                            │
│                                                                 │
│   Scalar<T>          Tensor<T,N>          Matrix<T,R,C>         │
│   (wraps dp::)       (wraps dp::)         (wraps dp::)          │
│                                                                 │
│   - operator+        - operator+          - operator+           │
│   - operator*        - dot(), norm()      - operator* (matmul)  │
│   - etc.             - etc.               - transpose(), etc.   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               │ uses internally
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EXPRESSION TEMPLATES                       │
│                      (optional, for lazy eval)                  │
│                                                                 │
│   BinaryAddOp<Tensor, Tensor>   - doesn't compute immediately   │
│   BinaryMulOp<Matrix, Matrix>   - evaluates on assignment       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               │ dispatches to
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         BACKEND                                 │
│                    (optimized kernels)                          │
│                                                                 │
│   backend/matmul.hpp    - _matmul<T,M,K,N>(a, b, c)             │
│   backend/transpose.hpp - _transpose<T,M,N>(in, out)            │
│   backend/reduce.hpp    - _sum<T,N>(data), _dot<T,N>(a,b)       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               │ uses
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SIMD INTRINSICS                            │
│                                                                 │
│   SIMDVec<float, 8>   - wraps __m256 (AVX)                      │
│   SIMDVec<double, 4>  - wraps __m256d (AVX)                     │
│   SIMDVec<float, 4>   - wraps __m128 (SSE)                      │
│                                                                 │
│   - load(), store()                                             │
│   - operator+, -, *, /                                          │
│   - hsum(), dot(), etc.                                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               │ needs
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         CONFIG                                  │
│                                                                 │
│   - Detect SSE/AVX/AVX2/AVX512/NEON                             │
│   - Define OPTINUM_HAS_SSE, OPTINUM_HAS_AVX, etc.               │
│   - Pick best SIMD width for platform                           │
└─────────────────────────────────────────────────────────────────┘
```

## Development Status

See [TODO.md](./TODO.md) for the complete development plan and current progress.

# Acknowledgements
 Made possible thanks to [those amazing projects](./ACKNOWLEDGMENTS.md).
