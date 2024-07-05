function downsample_image!(input_image::Matrix{Float64},
			pattern::Int64,
			downsampling_factor::Integer,
			mean_imgs_ij::Array{Float64})

	downsampled_image_size_x::Int64 = size(input_image)[1]/2
	downsampled_image_size_y::Int64 = size(input_image)[2]/2

	for j in 1:downsampled_image_size_y
		for i in 1:downsampled_image_size_x

 			im::Int64 = downsampling_factor*(i-1)+1
 			ip::Int64 = downsampling_factor*i
 			jm::Int64 = downsampling_factor*(j-1)+1
 			jp::Int64 = downsampling_factor*j
 			mean_imgs_ij[i, j, pattern] = sum(view(input_image, im:ip, jm:jp))

		end
	end
	return nothing
end


function get_widefield_images_ij!(obj_ij, 
				sub_patterns_ij,
				illuminated_obj_ij::Matrix{Float64},
				FFT_var::Matrix{ComplexF64},
				iFFT_var::Matrix{ComplexF64},
				img_ij::Matrix{ComplexF64},
				mean_img::Matrix{Float64},
				mean_imgs_ij::Array{Float64})
	

	for pattern in 1:n_patterns

		illuminated_obj_ij .= (obj_ij .* view(sub_patterns_ij, :, :, pattern))
 		ifftshift!(img_ij, illuminated_obj_ij)
   		mul!(FFT_var, fft_plan, img_ij)
 		FFT_var .= (FFT_var .* FFT_point_spread_function_ij)
 		mul!(iFFT_var, ifft_plan, FFT_var)
  	  	fftshift!(img_ij, iFFT_var)
 		mean_img .= abs.(real.(view(img_ij, 
 			half_padding_size+1:half_padding_size+2+padding_size,
 			half_padding_size+1:half_padding_size+2+padding_size)))
		downsample_image!(mean_img, pattern, 2, mean_imgs_ij)

	end


  	return nothing
end
