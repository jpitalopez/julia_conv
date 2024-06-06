##

using CUDA
using KernelAbstractions
using Pkg
using Images
using ImageView
##

# Load Image porbando commit
image = load("Lions_Glance_Snout_Black_and_white_552727_1920x1080.jpg")

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
  

    #Globa bidimensional index (x,y)
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
    
end

# Function call and time registrer
@time mul2_kernel(backend, 64)(IMAGE,MASC,OUTPUT,KERNEL_SIZE, IMAGE_SIZE;ndrange=size(IMAGE))


# Output
C = Array(OUTPUT)