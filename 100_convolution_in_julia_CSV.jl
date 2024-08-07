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

image = load("Lions.jpg")
#image = load("Grayscale_Cat.jpg")


# Load Kernel and convert to grayscale 
masc = load("mascara.png")
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


# Convolution function in device
@kernel function mul2_kernel(A,B,C,kernel_size, image_size)
  

    #Globa bidimensional index (x,y)patch_artist=True
    I= @index(Global,Cartesian)

    #Get padding unit
    a = (kernel_size[1]-1)/2
    pad_u = convert(Int, a)

    # Condition for always being inseide the Image
    if I[1] <= image_size[1]-pad_u && I[2] <= image_size[2]-pad_u && I[1]>pad_u && I[2]>pad_u

      sum = 0

      norm = 0

      for i in 1:kernel_size[1]

        for j in 1:kernel_size[2]
          
          sum = sum + B[i,j]*A[I[1]+i-pad_u-1,I[2]+j-pad_u-1]

          norm = norm + B[i,j]

        end

      end
      
      
      C[I[1],I[2]] = sum/norm

      
      
    end

    @synchronize 
  

end

##

# Function call and time registrer
CUDA.@elapsed mul2_kernel(backend, 64)(IMAGE,MASC,OUTPUT,KERNEL_SIZE, IMAGE_SIZE;ndrange=size(IMAGE))



##



# Num iterations
num_iteracions = 100

# Load times
times = Float64[]


# 100 iteractions Foor loop
for i in 1:num_iteracions
  
  
  # Call to the gpu function
  
  time_elapsed = CUDA.@elapsed mul2_kernel(backend, 64)(IMAGE,MASC,OUTPUT,KERNEL_SIZE, IMAGE_SIZE;ndrange=size(IMAGE))

  time_elapsed = time_elapsed * 1000
  
  
  
  # Add registrer to times vector
  push!(times, time_elapsed)

  
end

# Especifica la ruta donde quieres guardar el archivo CSV
csv_file = "times_29.csv"


df = DataFrame(times = times)

# Escribe el arreglo `times` en el archivo CSV usando CSV.write
CSV.write(csv_file, df)

# Compute mean and var
mean_times = mean(times)
var_times = var(times)

# Show results
println("Mean time: ", mean_times, " ms")
println("Var time: ", var_times, " ms")


##

# Output
C = Array(OUTPUT)

