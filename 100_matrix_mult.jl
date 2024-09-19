##

using Statistics
using CUDA
using KernelAbstractions
using Pkg
using Images
using ImageView
using CSV, DataFrames
##

# Load Image

image = load("matriz_aleatoria_1.png")
#image = load("Grayscale_Cat.jpg")


# Load Kernel and convert to grayscale 

masc = load("matriz_aleatoria_2.png")
masc = Gray.(masc)

# Load output
output = zeros(size(image))

# Create an array to save image dimensions
image_size = [size(image, 1) size(image, 2)]

# Create an array to save kernel dimensions
kernel_size = [size(masc, 1) size(masc, 2)]

##

# Move arrays to cuda device
IMAGE = CuArray(image)
MASC = CuArray(masc)
OUTPUT = CuArray(output)
OUTPUT = Gray.(OUTPUT)
IMAGE_SIZE = CuArray(image_size)
KERNEL_SIZE = CuArray(kernel_size)

##

# Get backend
backend = get_backend(IMAGE)


@kernel function matrix_multiply_kernel(A, B, C, kernel_size, image_size)
    
  # Obtener los índices globales en la cuadrícula 2D
  I = @index(Global, Cartesian)

  # Inicializar la suma para cada elemento de C
  sum = 0.0

  # Obtener las dimensiones de la matriz A
  N = kernel_size[1]  # Asumimos que kernel_size es el tamaño de las matrices (cuadradas)

  # Asegurarse de que los índices no superen el tamaño de la matriz
  if I[1] <= image_size[1] && I[2] <= image_size[2]
      # Realizar el producto escalar de la fila de A y la columna de B
      for k in 1:N
          sum += A[I[1], k] * B[k, I[2]]
      end

      # Guardar el resultado en la matriz C
      C[I[1], I[2]] = sum
  end

  @synchronize

end


# Function call and time registrer
time = @CUDA.elapsed matrix_multiply_kernel(backend, 32)(IMAGE,MASC,OUTPUT,KERNEL_SIZE, IMAGE_SIZE;ndrange=size(IMAGE))

# Num iterations
num_iteracions = 100

# Load times
times = Float64[]


# 100 iteractions Foor loop
for i in 1:num_iteracions
    
  # Call to the gpu function
  
  time_elapsed = CUDA.@elapsed matrix_multiply_kernel(backend, 32)(IMAGE,MASC,OUTPUT,KERNEL_SIZE, IMAGE_SIZE;ndrange=size(IMAGE))

  time_elapsed = time_elapsed * 1000
    
  # Add registrer to times vector
  push!(times, time_elapsed)

  
end

# Especifica la ruta donde quieres guardar el archivo CSV
csv_file = "multiply_times_1000.csv"

df = DataFrame(times = times)

# Escribe el arreglo `times` en el archivo CSV usando CSV.write
CSV.write(csv_file, df)

# Compute mean and var
mean_times = mean(times)
var_times = var(times)

# Show results
println("Mean time: ", mean_times, " ms")
println("Var time: ", var_times, " ms")

C = Array(OUTPUT)