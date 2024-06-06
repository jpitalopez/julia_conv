##

using CUDA
using KernelAbstractions
using Pkg
using Images
using ImageView
##

# Load Image
a = load("Lions_Glance_Snout_Black_and_white_552727_1920x1080.jpg")

# Load Kernel and convert to grayscale 
b = load("mascara.png")
b = Gray.(b)

# Load output
c = zeros(size(a))

# Create an array to save dimensions
e = [size(b, 1) size(b, 1)]

##


# Move arrays to cuda device
A = CuArray(a)
B = CuArray(b)
C = CuArray(c)
C = Gray.(C)
E = CuArray(e)

##

# Get backend
backend = get_backend(A)


# Convolution function in device
@kernel function mul2_kernel(A,B,C,kernel_size)
  

    #Globa bidimensional index (x,y)
    I= @index(Global,Cartesian)

    #Get padding unit
    a = (kernel_size[1]-1)/2
    pad_u = convert(Int, a)

    # Condition for always being inseide the Image
    if I[1] <= 1080-pad_u && I[2] <= 1920-pad_u && I[1]>pad_u && I[2]>pad_u

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
@time mul2_kernel(backend, 64)(A,B,C,E;ndrange=size(A))


# Output
C = Array(C)