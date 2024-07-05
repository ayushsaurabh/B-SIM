function downsample_image!(input_image::Matrix{Float64},
			pattern::Int64,
			downsampling_factor::Integer,
			mean_images::Array{Float64})

	downsampled_image_size_x::Int64 = size(input_image)[1]/2
	downsampled_image_size_y::Int64 = size(input_image)[2]/2

	for j in 1:downsampled_image_size_y
		for i in 1:downsampled_image_size_x

 			im::Int64 = downsampling_factor*(i-1)+1
 			ip::Int64 = downsampling_factor*i
 			jm::Int64 = downsampling_factor*(j-1)+1
 			jp::Int64 = downsampling_factor*j
 			mean_images[i, j, pattern] = sum(view(input_image, im:ip, jm:jp))
		end
	end
	return nothing
end


function get_widefield_images!(object::Matrix{Float64},
				illuminated_object::Matrix{Float64},
				FFT_var::Matrix{ComplexF64},
				iFFT_var::Matrix{ComplexF64},
				img::Matrix{ComplexF64},
				mean_img::Matrix{Float64},
				mean_images::Array{Float64})

	for pattern in 1:n_patterns

		illuminated_object .= (object .* view(patterns_with_padding, :, :, pattern))
		ifftshift!(img, illuminated_object)
 		mul!(FFT_var, fft_plan_global, img)
		FFT_var .= (FFT_var .* FFT_point_spread_function)
		mul!(iFFT_var, ifft_plan_global, FFT_var)
 	  	fftshift!(img, iFFT_var)
		mean_img .= abs.(real.(img))
 		downsample_image!(mean_img, pattern, 2, mean_images)

	end

	return nothing
end
