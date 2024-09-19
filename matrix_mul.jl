using CUDA
using KernelAbstractions
using Pkg
using Images
using ImageView





matrix_1 = load("matriz_aleatoria_1.png")
#image = load("Grayscale_Cat.jpg")

matrix_2 = load("matriz_aleatoria_2.png")

a = CuArray(matrix_1)
b = CuArray(matrix_2)

M, N, K = 15, 15, 15

output = CuArray(zeros(M, N))

output = a*b

backend = get_backend(a)


# Simple kernel for matrix multiplication
@kernel function matmul_kernel!(output, a, b)
    i, j = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(output))
    for k in 1:size(a)[2]
        tmp_sum += a[i, k] * b[k, j]
    end

    output[i, j] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function matmul!(output, a, b)
    if size(a)[2] != size(b)[1]
        println("Matrix size mismatch!")
        return nothing
    end
    backend = KernelAbstractions.get_backend(a)
    kernel! = matmul_kernel!(backend)
    kernel!(output, a, b, ndrange = size(output))
end

matmul!(output, a, b)
KernelAbstractions.synchronize(backend)

print(output)