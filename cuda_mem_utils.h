#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#define CUDA_CHECK(call)                                                      \
    do { cudaError_t _e = (call);                                             \
         if (_e != cudaSuccess) {                                             \
             fprintf(stderr,"CUDA error %s:%d  %s\n",                         \
                     __FILE__,__LINE__,cudaGetErrorString(_e));               \
             std::abort();                                                    \
         }                                                                    \
    } while (0)

enum class InitMode { NONE, ZERO, NEG_ONE };

template<typename T>
T* cuda_alloc(std::size_t n, InitMode mode = InitMode::NONE)
{
    if (n == 0) return nullptr;

    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));

    switch (mode)
    {
    case InitMode::ZERO:
        CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(T)));
        break;

    case InitMode::NEG_ONE:
        CUDA_CHECK(cudaMemset(ptr, 0xFFFFFFFF, n * sizeof(T)));
        break;

    case InitMode::NONE:
    default:
        break;
    }
    return ptr;
}

template<typename T>
void cuda_free(T*& p)
{
    if (p) CUDA_CHECK(cudaFree(p));
    p = nullptr;
}

#define CUDA_ALLOC(PTR, N, MODE)   (PTR) = cuda_alloc<std::remove_pointer_t<decltype(PTR)>>(N, MODE)
#define CUDA_FREE(PTR)             cuda_free(PTR)

enum class CopyDir { H2D, D2H, D2D };

template<typename T>
inline void cuda_copy(T* dst, const T* src, std::size_t n, CopyDir dir)
{
    if (n == 0) return;
    cudaMemcpyKind k =
        (dir == CopyDir::H2D) ? cudaMemcpyHostToDevice :
        (dir == CopyDir::D2H) ? cudaMemcpyDeviceToHost :
        cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(T), k));
}