##


using CUDA
using KernelAbstractions
using Pkg
using Images
using ImageView
##


a = load("Lions_Glance_Snout_Black_and_white_552727_1920x1080.jpg")
b = load("mascara.png")
b = Gray.(b)

d = [0 1 0; 1 4 1; 0 1 0]
c = zeros(size(a))




e = [91 91]

##


A = CuArray(a)
B = CuArray(b)
C = CuArray(c)
C = Gray.(C)
D = CuArray(d)
E = CuArray(e)


##


backend = get_backend(A)


@kernel function mul2_kernel(A,B,C,D,kernel_size)

    I= @index(Global,Cartesian)


    pad_u = 45


    if I[1] <= 1080-pad_u && I[2] <= 1920-pad_u && I[1]>pad_u && I[2]>pad_u

      sum = 0

      norm = 0



      for i in 1:kernel_size[1]

        for j in 1:kernel_size[2]


          sum = sum + D[i,j]*A[I[1]+i-pad_u-1,I[2]+j-pad_u-1]

          norm = norm + D[i,j]
   



        end

      end
      

      C[I[1],I[2]] = sum/norm

    end
  
  
end




@time mul2_kernel(backend, 64)(A,B,C,B,E;ndrange=size(A))



C = Array(C)



