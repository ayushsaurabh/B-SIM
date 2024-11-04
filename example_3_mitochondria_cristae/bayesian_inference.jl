function log_Beta(a::Vector{Float64})
	z::Float64 = sum(loggamma.(a)) - loggamma(sum(a))
	return z
end

const log_Beta_MTF::Float64= log_Beta(modulation_transfer_function_sim_ij_vectorized .+ eps())
const MTF_minus_one::Vector{Float64} = (conc_parameter .* modulation_transfer_function_sim_ij_vectorized) .- 1.0


function log_Dirichlet(x::Vector{Float64}, a::Vector{Float64})
	z::Float64 = sum( (a .- 1.0)  .* log.(x)) - log_Beta(a)
	return z
end
function log_Dirichlet_MTF(x::Vector{Float64})
	x .= MTF_minus_one .* log.(x)
	z::Float64 = sum(x) - log_Beta_MTF
	return z
end

function log_Poisson(x::Float64, lambda::Float64)
	return x*log(lambda) - lambda - loggamma(x+1.0)
end

const log_2pi::Float64 = log(2.0*pi) 
function log_Normal(x::Float64, mu_normal::Float64, sigma_normal::Float64)
	return -((x - mu_normal)^2)/(2.0 * sigma_normal^2) - 0.5*(log_2pi + 2.0*log(sigma_normal))
end


function get_log_likelihood_ij(ii::Int64, jj::Int64,
				mean_imgs_ij::Array{Float64},
				shot_noise_imgs_ij,
				bg_ij)

	i_minus::Int64 = 1
	i_plus::Int64 = half_padding_size+1

#	if (i_procs== 0) && (0 < ii - half_padding_size <= quarter_padding_size)
#
#    		i_minus += (quarter_padding_size - (ii - half_padding_size - 1))
#
#	end
#	if (i_procs==n_procs_per_dim_x-1) &&
#		(0 <= -((ii - half_padding_size) - sub_raw_img_size_x)  < quarter_padding_size)
#
# 		i_plus -= (quarter_padding_size + ((ii - half_padding_size) - sub_raw_img_size_x))
#
#	end

	j_minus::Int64 = 1
	j_plus::Int64 = half_padding_size+1
#	if (j_procs== 0) && (0 < jj - half_padding_size <= quarter_padding_size)
#
#  		j_minus += (quarter_padding_size - (jj - half_padding_size - 1))
#
#	end
#	if (j_procs==n_procs_per_dim_y-1) &&
#		(0 <= -((jj - half_padding_size) - sub_raw_img_size_y) < quarter_padding_size)
#
# 		j_plus -= (quarter_padding_size + ((jj - half_padding_size) - sub_raw_img_size_y))
#
#	end

 	log_likelihood::Float64 = 0.0 
	for pattern in 1:n_patterns
 		for j in j_minus:j_plus
 			for i in i_minus:i_plus
				log_likelihood += log_Poisson(shot_noise_imgs_ij[i, j, pattern], 
							      mean_imgs_ij[i, j, pattern]+bg_ij[i, j]+ eps())
 			end
 		end
	end


	return log_likelihood 
end


function get_log_prior_ij!(FFT_var::Matrix{ComplexF64},
			img_ij::Matrix{ComplexF64},
			img_ij_abs::Matrix{Float64},
			mod_fft_img_ij::Vector{Float64},
			i::Int64, j::Int64)

   	fftshift!(img_ij, FFT_var)
	img_ij_abs .= abs.(img_ij) 
	mod_fft_img_ij .= (vec(img_ij_abs) ./ sum(img_ij_abs)) .+ eps()

 	return log_Dirichlet_MTF(mod_fft_img_ij)
end

# The following Gibbs sampler computes likelihood for the neighborhood only.
function sample_object_neighborhood_MLE!(temperature::Float64,
				object::Matrix{Float64},
				shot_noise_images::Array{Float64},
				bg::Matrix{Float64},
				illuminated_obj_ij::Matrix{Float64},
				FFT_var::Matrix{ComplexF64},
				iFFT_var::Matrix{ComplexF64},
				img_ij::Matrix{ComplexF64},
				img_ij_abs::Matrix{Float64},
				mean_img::Matrix{Float64},
				mean_imgs_ij::Array{Float64},
				proposed_mean_imgs_ij::Array{Float64},
				mod_fft_img_ij::Vector{Float64},
				old_values::Matrix{Float64},
				proposed_values::Matrix{Float64})
	n_accepted::Int64 = 0
	for j in padding_size+1:2:padding_size+sub_sim_img_size_y
		for i in padding_size+1:2:padding_size+sub_sim_img_size_x

			ii::Int64 = ceil(i/2)
			jj::Int64 = ceil(j/2)

 			old_values .= object[i:i+1, j:j+1]
 
 
 			shot_noise_imgs_ij = view(shot_noise_images, 
     					ii - quarter_padding_size:ii + quarter_padding_size, 
     					jj - quarter_padding_size:jj + quarter_padding_size, :)

 			bg_ij = view(bg, 
     					ii - quarter_padding_size:ii + quarter_padding_size, 
     					jj - quarter_padding_size:jj + quarter_padding_size)

 
 			obj_ij = view(object, 
   					i - padding_size:i+1 + padding_size, 
   					j - padding_size:j+1 + padding_size)
 
 			sub_patterns_ij = view(sub_patterns, 
   					i - padding_size:i+1 + padding_size, 
   					j - padding_size:j+1 + padding_size, :)
 
 			get_widefield_images_ij!(obj_ij, 
 						sub_patterns_ij, 
  						illuminated_obj_ij,
      						FFT_var,
      						iFFT_var,
      						img_ij,
  						mean_img,
  						mean_imgs_ij) 

 
 			old_log_likelihood::Float64 = get_log_likelihood_ij(i, j,
  							mean_imgs_ij, shot_noise_imgs_ij, bg_ij)
 
#       		ifftshift!(img_ij, obj_ij)
#    			mul!(FFT_var, fft_plan, img_ij)
#  			old_log_prior::Float64 = 
#       				get_log_prior_ij!(FFT_var, 
#     						img_ij, 
#      						img_ij_abs, 
#      						mod_fft_img_ij, 
#      						i, j)  
 			old_jac::Float64 = sum(log.(old_values))
  			old_log_posterior::Float64 = old_log_likelihood #+ old_log_prior
 			
  			proposed_values .= rand.(Normal.(log.(old_values), 
 							covariance_object))
 			proposed_values .= exp.(proposed_values)
  			object[i:i+1, j:j+1] .= proposed_values
 
 			proposed_obj_ij = view(object, 
     				i - padding_size:i+1 + padding_size, 
     				j - padding_size:j+1 + padding_size)
 
 			get_widefield_images_ij!(proposed_obj_ij, sub_patterns_ij, 
  						illuminated_obj_ij,
      						FFT_var,
      						iFFT_var,
      						img_ij,
  						mean_img,
  						proposed_mean_imgs_ij) 

 						
 			proposed_log_likelihood::Float64 = get_log_likelihood_ij(i, j,
  							proposed_mean_imgs_ij, shot_noise_imgs_ij, bg_ij)
 
#         		ifftshift!(img_ij, proposed_obj_ij)
#      			mul!(FFT_var, fft_plan, img_ij)
#  			proposed_log_prior::Float64 = 
#         				get_log_prior_ij!(FFT_var, 
#       							img_ij, 
#        						img_ij_abs, 
#        						mod_fft_img_ij, 
#        						i, j)  
# 
  			proposed_jac::Float64 = sum(log.(proposed_values))
  			proposed_log_posterior::Float64 = proposed_log_likelihood #+ proposed_log_prior
 
   			log_hastings::Float64 = (1.0/temperature) *
                           	(proposed_log_posterior - old_log_posterior) +
    							proposed_jac - old_jac
 			log_rand::Float64 = log(rand(rng, Float64))
   
    			if log_hastings > 0 #log_rand
  				copy!(mean_imgs_ij, proposed_mean_imgs_ij)
    				n_accepted += 1
				old_log_likelihood = proposed_log_likelihood
 			else
 				object[i:i+1, j:j+1] .= old_values
 			end

			# Sample Background

			old_value::Float64 = bg[ii, jj]
			old_log_posterior = old_log_likelihood

			proposed_value::Float64 = rand(Normal(log(old_value), 
							   covariance_object))[1]
 			proposed_value = exp(proposed_value)
			bg[ii, jj] = proposed_value

 			proposed_bg_ij = view(bg, 
     					ii - quarter_padding_size:ii + quarter_padding_size, 
     					jj - quarter_padding_size:jj + quarter_padding_size)

 			obj_ij = view(object, 
   					i - padding_size:i+1 + padding_size, 
   					j - padding_size:j+1 + padding_size)

  			proposed_log_likelihood = get_log_likelihood_ij(i, j,
  							mean_imgs_ij, shot_noise_imgs_ij, proposed_bg_ij)

  			proposed_log_posterior = proposed_log_likelihood 
 
			old_jac = log(old_value)
  			proposed_jac = log(proposed_value)
   			log_hastings = (1.0/temperature) *
                           	(proposed_log_posterior - old_log_posterior) +
    							proposed_jac - old_jac
 			log_rand = log(rand(rng, Float64))
   
    			if log_hastings > 0 #log_rand
    				n_accepted += 1
 			else
 				bg[ii, jj] = old_value
 			end


			# Sample Intermediate Expected Photon Counts on each Pixel
			# Choose the central pixel in the mean image
			# for expected photon count

 			for pattern in 1:n_patterns

   				expected_photon_count::Float64 = mean_imgs_ij[
   							quarter_padding_size+1, 
							quarter_padding_size+1, pattern] + bg[ii, jj]
   
     				old_log_likelihood = log_Normal(sub_raw_images[ii, jj, pattern],
							sub_gain_map[ii, jj]*shot_noise_images[ii, jj, pattern] +
  							sub_offset_map[ii, jj],
  							sub_error_map[ii, jj] + eps())
   
 
  				old_log_prior = log_Poisson(shot_noise_images[ii, jj, pattern],
							    expected_photon_count + eps()) 
  				old_log_posterior = old_log_likelihood  + old_log_prior
    
     				proposed_shot_noise_pixel::Float64 =
  					rand(rng, Poisson(shot_noise_images[ii, jj, pattern] + eps()), 1)[1]
   
     				proposed_log_likelihood = log_Normal(sub_raw_images[ii, jj, pattern], 
							sub_gain_map[ii, jj]*proposed_shot_noise_pixel +
     							sub_offset_map[ii, jj], 
    							sub_error_map[ii, jj] + eps())
   
   				proposed_log_prior = log_Poisson(proposed_shot_noise_pixel,
							    expected_photon_count + eps()) 
  				proposed_log_posterior = proposed_log_likelihood  + proposed_log_prior
   
   				log_forward_proposal_probability::Float64 =
						log_Poisson(proposed_shot_noise_pixel,
							shot_noise_images[ii, jj, pattern]+ eps()) 

 				log_backward_proposal_probability::Float64 = 
 						log_Poisson(shot_noise_images[ii, jj, pattern],
							proposed_shot_noise_pixel + eps()) 
    
     				log_hastings = (1.0/temperature)*
             				(proposed_log_posterior - old_log_posterior) +
   						(log_backward_proposal_probability -
   						log_forward_proposal_probability)
    
  				log_rand = log(rand(rng, Float64))
    				if log_hastings > 0 #log_rand 
     					shot_noise_images[ii, jj, pattern] = proposed_shot_noise_pixel
     				end
 			end

		end
	end

	return n_accepted
end
function sample_object_neighborhood_MCMC!(temperature::Float64,
				object::Matrix{Float64},
				shot_noise_images::Array{Float64},
				bg::Matrix{Float64},
				illuminated_obj_ij::Matrix{Float64},
				FFT_var::Matrix{ComplexF64},
				iFFT_var::Matrix{ComplexF64},
				img_ij::Matrix{ComplexF64},
				img_ij_abs::Matrix{Float64},
				mean_img::Matrix{Float64},
				mean_imgs_ij::Array{Float64},
				proposed_mean_imgs_ij::Array{Float64},
				mod_fft_img_ij::Vector{Float64},
				old_values::Matrix{Float64},
				proposed_values::Matrix{Float64})
	n_accepted::Int64 = 0
	for j in padding_size+1:2:padding_size+sub_sim_img_size_y
		for i in padding_size+1:2:padding_size+sub_sim_img_size_x

			ii::Int64 = ceil(i/2)
			jj::Int64 = ceil(j/2)

 			old_values .= object[i:i+1, j:j+1]
 
 
 			shot_noise_imgs_ij = view(shot_noise_images, 
     					ii - quarter_padding_size:ii + quarter_padding_size, 
     					jj - quarter_padding_size:jj + quarter_padding_size, :)

 			bg_ij = view(bg, 
     					ii - quarter_padding_size:ii + quarter_padding_size, 
     					jj - quarter_padding_size:jj + quarter_padding_size)

 
 			obj_ij = view(object, 
   					i - padding_size:i+1 + padding_size, 
   					j - padding_size:j+1 + padding_size)
 
 			sub_patterns_ij = view(sub_patterns, 
   					i - padding_size:i+1 + padding_size, 
   					j - padding_size:j+1 + padding_size, :)
 
 			get_widefield_images_ij!(obj_ij, 
 						sub_patterns_ij, 
  						illuminated_obj_ij,
      						FFT_var,
      						iFFT_var,
      						img_ij,
  						mean_img,
  						mean_imgs_ij) 

 
 			old_log_likelihood::Float64 = get_log_likelihood_ij(i, j,
  							mean_imgs_ij, shot_noise_imgs_ij, bg_ij)
 
#       			ifftshift!(img_ij, obj_ij)
#    			mul!(FFT_var, fft_plan, img_ij)
#  			old_log_prior::Float64 = 
#       				get_log_prior_ij!(FFT_var, 
#     						img_ij, 
#      						img_ij_abs, 
#      						mod_fft_img_ij, 
#      						i, j)  
 			old_jac::Float64 = sum(log.(old_values))
  			old_log_posterior::Float64 = old_log_likelihood #+ old_log_prior
 			
  			proposed_values .= rand.(Normal.(log.(old_values), 
 							covariance_object))
 			proposed_values .= exp.(proposed_values)
  			object[i:i+1, j:j+1] .= proposed_values
 
 			proposed_obj_ij = view(object, 
     				i - padding_size:i+1 + padding_size, 
     				j - padding_size:j+1 + padding_size)
 
 			get_widefield_images_ij!(proposed_obj_ij, sub_patterns_ij, 
  						illuminated_obj_ij,
      						FFT_var,
      						iFFT_var,
      						img_ij,
  						mean_img,
  						proposed_mean_imgs_ij) 

 						
 			proposed_log_likelihood::Float64 = get_log_likelihood_ij(i, j,
  							proposed_mean_imgs_ij, shot_noise_imgs_ij, bg_ij)
 
#         		ifftshift!(img_ij, proposed_obj_ij)
#      			mul!(FFT_var, fft_plan, img_ij)
#  			proposed_log_prior::Float64 = 
#         				get_log_prior_ij!(FFT_var, 
#       							img_ij, 
#        						img_ij_abs, 
#        						mod_fft_img_ij, 
#        						i, j)  
# 
  			proposed_jac::Float64 = sum(log.(proposed_values))
  			proposed_log_posterior::Float64 = proposed_log_likelihood #+ proposed_log_prior
 
   			log_hastings::Float64 = (1.0/temperature) *
                           	(proposed_log_posterior - old_log_posterior) +
    							proposed_jac - old_jac
 			log_rand::Float64 = log(rand(rng, Float64))
   
    			if log_hastings > log_rand
  				copy!(mean_imgs_ij, proposed_mean_imgs_ij)
    				n_accepted += 1
				old_log_likelihood = proposed_log_likelihood
 			else
 				object[i:i+1, j:j+1] .= old_values
 			end

			# Sample Background

			old_value::Float64 = bg[ii, jj]
			proposed_value::Float64 = rand(Normal(log(old_value), 
							   covariance_object))[1]
 			proposed_value = exp(proposed_value)
			bg[ii, jj] = proposed_value

 			proposed_bg_ij = view(bg, 
     					ii - quarter_padding_size:ii + quarter_padding_size, 
     					jj - quarter_padding_size:jj + quarter_padding_size)

 
 			obj_ij = view(object, 
   					i - padding_size:i+1 + padding_size, 
   					j - padding_size:j+1 + padding_size)

  			proposed_log_likelihood = get_log_likelihood_ij(i, j,
  							mean_imgs_ij, shot_noise_imgs_ij, proposed_bg_ij)

  			proposed_log_posterior = proposed_log_likelihood #+ proposed_log_prior
 
			old_jac = log(old_value)
  			proposed_jac = log(proposed_value)
   			log_hastings = (1.0/temperature) *
                           	(proposed_log_posterior - old_log_posterior) +
    							proposed_jac - old_jac
 			log_rand = log(rand(rng, Float64))
   
    			if log_hastings > log_rand
    				n_accepted += 1
 			else
 				bg[ii, jj] = old_value
 			end


			# Sample Intermediate Expected Photon Counts on each Pixel
			# Choose the central pixel in the mean image
			# for expected photon count

 			for pattern in 1:n_patterns

   				expected_photon_count::Float64 = mean_imgs_ij[
   							quarter_padding_size+1, 
							quarter_padding_size+1, pattern] + bg[ii, jj]
   
     				old_log_likelihood = log_Normal(sub_raw_images[ii, jj, pattern],
							sub_gain_map[ii, jj]*shot_noise_images[ii, jj, pattern] +
  							sub_offset_map[ii, jj],
  							sub_error_map[ii, jj] + eps())
   
 
  				old_log_prior = log_Poisson(shot_noise_images[ii, jj, pattern],
							    expected_photon_count + eps()) 
  				old_log_posterior = old_log_likelihood  + old_log_prior
    
     				proposed_shot_noise_pixel::Float64 =
  					rand(rng, Poisson(shot_noise_images[ii, jj, pattern] + eps()), 1)[1]
   
     				proposed_log_likelihood = log_Normal(sub_raw_images[ii, jj, pattern], 
							sub_gain_map[ii, jj]*proposed_shot_noise_pixel +
     							sub_offset_map[ii, jj], 
    							sub_error_map[ii, jj] + eps())
   
   				proposed_log_prior = log_Poisson(proposed_shot_noise_pixel,
							    expected_photon_count + eps()) 
  				proposed_log_posterior = proposed_log_likelihood  + proposed_log_prior
   
   				log_forward_proposal_probability::Float64 =
						log_Poisson(proposed_shot_noise_pixel,
							shot_noise_images[ii, jj, pattern]+ eps()) 

 				log_backward_proposal_probability::Float64 = 
 						log_Poisson(shot_noise_images[ii, jj, pattern],
							proposed_shot_noise_pixel + eps()) 
    
     				log_hastings = (1.0/temperature)*
             				(proposed_log_posterior - old_log_posterior) +
   						(log_backward_proposal_probability -
   						log_forward_proposal_probability)
    
  				log_rand = log(rand(rng, Float64))
    				if log_hastings > log_rand 
     					shot_noise_images[ii, jj, pattern] = proposed_shot_noise_pixel
     				end
 			end

		end
	end

	return n_accepted

end

