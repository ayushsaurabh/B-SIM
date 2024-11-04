function log_Beta(a::Vector{Float64})
	z::Float64 = sum(loggamma.(a)) - loggamma(sum(a))
	return z
end

const log_Beta_MTF::Float64= log_Beta(modulation_transfer_function_sim_vectorized .+ eps())
const MTF_minus_one::Vector{Float64} = (conc_parameter .* modulation_transfer_function_sim_vectorized) .- 1.0


function log_Dirichlet(x::Vector{Float64}, a::Vector{Float64})
	z::Float64 = sum( (a .- 1.0)  .* log.(x)) - log_Beta(a)
	return z
end
function log_Dirichlet_MTF(x::Vector{Float64})
	x .= MTF_minus_one .* log.(x)
	z::Float64 = sum(x) - log_Beta_MTF
	return z
end

function get_log_prior!(FFT_var::Matrix{ComplexF64},
			img::Matrix{ComplexF64},
			img_abs::Matrix{Float64},
			mod_fft_img::Vector{Float64})

   	fftshift!(img, FFT_var)
	img_abs .= abs.(img) 
	mod_fft_img .= (vec(img_abs) ./ sum(img_abs)) .+ eps()

 	return log_Dirichlet_MTF(mod_fft_img)
end


function get_log_likelihood(object::Matrix{Float64},
			shot_noise_images::Array{Float64},							
			bg::Matrix{Float64},
			illuminated_object::Matrix{Float64},
			FFT_var::Matrix{ComplexF64},
			iFFT_var::Matrix{ComplexF64},
			img::Matrix{ComplexF64},
			mean_img::Matrix{Float64},
			mean_images::Array{Float64})



	log_likelihood::Float64 = 0.0
	for pattern in 1:n_patterns
		log_likelihood += sum(logpdf.(Poisson.(
				view(mean_images, half_padding_size+1:half_padding_size+raw_img_size_x, 
				     half_padding_size+1:half_padding_size+raw_img_size_y, pattern) .+ 
				view(bg, half_padding_size+1:half_padding_size+raw_img_size_x, 
				     half_padding_size+1:half_padding_size+raw_img_size_y) .+ eps()),
				view(shot_noise_images, half_padding_size+1:half_padding_size+raw_img_size_x, 
					half_padding_size+1:half_padding_size+raw_img_size_y, pattern)))

		log_likelihood += sum(logpdf.(Normal.(
                		(view(gain_map_with_padding, half_padding_size+1:half_padding_size+raw_img_size_x, 
				     	half_padding_size+1:half_padding_size+raw_img_size_y) .*
				view(shot_noise_images, half_padding_size+1:half_padding_size+raw_img_size_x, 
				    	half_padding_size+1:half_padding_size+raw_img_size_y, pattern)) .+
				view(offset_map_with_padding, half_padding_size+1:half_padding_size+raw_img_size_x, 
				    	half_padding_size+1:half_padding_size+raw_img_size_y),
                    		view(error_map_with_padding, half_padding_size+1:half_padding_size+raw_img_size_x, 
			 		half_padding_size+1:half_padding_size+raw_img_size_y) .+ eps()),
				view(raw_images_with_padding, half_padding_size+1:half_padding_size+raw_img_size_x, 
			     		half_padding_size+1:half_padding_size+raw_img_size_y, pattern)))
	end

	return log_likelihood
end

function compute_full_log_posterior!(object::Matrix{Float64},
				shot_noise_images::Array{Float64},
				bg::Matrix{Float64},
				illuminated_object::Matrix{Float64},
				FFT_var::Matrix{ComplexF64},
				iFFT_var::Matrix{ComplexF64},
				img::Matrix{ComplexF64},
				img_abs::Matrix{Float64},
				mean_img::Matrix{Float64},
				mean_images::Array{Float64},
				mod_fft_img::Vector{Float64})

	get_widefield_images!(object,
			illuminated_object::Matrix{Float64},
			FFT_var::Matrix{ComplexF64},
			iFFT_var::Matrix{ComplexF64},
			img::Matrix{ComplexF64},
			mean_img::Matrix{Float64},
			mean_images::Array{Float64})


	log_likelihood::Float64 = get_log_likelihood(object, 
						shot_noise_images,
						bg,
						illuminated_object,
						FFT_var,
						iFFT_var,
						img,
						mean_img,
						mean_images)
# 	ifftshift!(img, object)
#   	mul!(FFT_var, fft_plan_global, img)
# 	log_prior::Float64 = get_log_prior!(FFT_var, 
#   						img, 
#    						img_abs, 
#    						mod_fft_img)  

	log_posterior::Float64 = log_likelihood #+ log_prior

  	@show log_posterior


	return log_posterior
end

