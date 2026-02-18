module CUDAExt

using CrystallographicFFT
using CUDA

# In modern CUDA.jl, CUDABackend is re-exported from CUDA
# We use the backend from CUDA directly
CrystallographicFFT.CFFTApi._infer_backend(::Type{<:CuArray}) = CUDA.CUDABackend()
CrystallographicFFT.CFFTApi._infer_backend(::Type{CuArray{T,N}}) where {T,N} = CUDA.CUDABackend()

end # module CUDAExt
