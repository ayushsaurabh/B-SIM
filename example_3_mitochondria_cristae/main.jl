# B-SIM

###########################################################
# Copyright (C) 2022 Presse Lab - All Rights Reserved
#
# Author: Ayush Saurabh
#
# You may use, distribute and modify this code under the
# terms of the MIT license.
###########################################################

###############################################################################
# A brief description of the sampler:
# The sampler here execute a Markov Chain Monte Carlo (MCMC) algorithm (Gibbs)
# where samples for fluorescence intensity at each pixel are generated
# sequentially from their corresponding conditional probability distributions
# (posterior). First, the sampler creates/initiates arrays to store all the
# samples and joint posterior probability values. Next, new samples are then
# iteratively proposed using proposal (normal) distributions for each pixel, to
# be accepted or rejected by the Metropolis-Hastings step if direct sampling is
# not available. If accepted, the proposed sample is stored in the arrays
# otherwise the previous sample is stored at the same MCMC iteraion.
#
# The variance of the proposal distribution typically decides how often
# proposals are accepted/rejected. A larger covariance or movement away from
# the previous sample would lead to a larger change in likelihood/posterior
# values.  Since the sampler prefers movement towards high probability regions,
# a larger movement towards low probability regions would lead to likely
# rejection of the sample compared to smaller movement.
#
# The collected samples can then be used to compute statistical quantities and
# plot probability distributions. The plotting function used by the sampler in
# this code allows monitoring of posterior values and current fluorescence
# intensity sample.

# As mentioned before, sampler prefers movement towards higher probability
# regions of the posterior distribution. This means that if parameters are
# initialized in low probability regions of the posterior, which is typically
# the case, the posterior would appear to increase initially for many iterations
# (hundreds to thousands depending on the complexity of the model). This initial
# period is called burn-in. After burn-in, convergence is achieved where the
# posterior would typically fluctuate around some mean/average value. The
# convergence typically indicates that the sampler has reached the maximum of
# the posterior distribution (the most probable region), that is, sampler
# generates most samples from higher probability region. In other words, given a
# large collection of samples, the probability density in a region of parameter
# space is proportional to the number of samples collected from that region.
#
# All the samples collected during burn-in are usually ignored when computing
# statistical properties and presenting the final posterior distribution.
###############################################################################

using TiffImages
using Statistics
using Colors

include("input_parameters.jl")
include("input_data.jl")

using Distributed

println("Adding processors...")
flush(stdout);
addprocs(n_procs_per_dim_x*n_procs_per_dim_y, topology=:master_worker)
println("Done.")
flush(stdout);

@everywhere using Random, Distributions, SpecialFunctions, LinearAlgebra, FFTW
@everywhere rng = MersenneTwister(myid());

if plotting == true

	println("Plotting is On.")
	flush(stdout);

	using Plots
else

	println("Plotting is Off.")
	flush(stdout);

end


include("psf.jl")
include("image_formation.jl")
include("global_posterior.jl")
include("output.jl")

@everywhere workers() begin

	include("input_parameters.jl")
	include("chunk_images.jl")
 	include("psf_for_inference.jl")
 	include("image_formation_for_inference.jl")
 	include("bayesian_inference.jl")

end

function sampler()

 	@show psf_type
 	@show abbe_diffraction_limit
 	@show physical_pixel_size
 	@show padding_size
 	@show median_photon_count

	# Initialize
	draw::Int64 = 1
	println("draw = ", draw)
	flush(stdout);

	# Arrays for main variables of interest
	object::Matrix{Float64} = zeros(sim_img_size_x+2*padding_size, sim_img_size_y+2*padding_size)
 	object[padding_size+1:end-padding_size,
 	       padding_size+1:end-padding_size] .= rand(rng, sim_img_size_x, sim_img_size_y)
	intermediate_object::Matrix{Float64} = zeros(3*sim_img_size_x, 3*sim_img_size_y)
	apply_reflective_BC_object!(object, intermediate_object)

	sum_object::Matrix{Float64} =
		zeros(sim_img_size_x+2*padding_size, sim_img_size_y+2*padding_size)
	mean_object::Matrix{Float64} =
	 	zeros(sim_img_size_x+2*padding_size, sim_img_size_y+2*padding_size)
	
	shot_noise_images::Array{Float64} = 
		zeros(raw_img_size_x+2*half_padding_size, raw_img_size_y+2*half_padding_size, n_patterns)
	shot_noise_images[half_padding_size+1:end-half_padding_size, 
    		half_padding_size+1:end-half_padding_size, :] .= 
 			Int64.(ceil.(abs.((input_raw_images .- median(offset_map_with_padding)) ./ 
					  (median(gain_map_with_padding)))))
	intermediate_shot::Array{Float64} = 
		zeros(3*raw_img_size_x, 3*raw_img_size_y, n_patterns)
	apply_reflective_BC_shot!(shot_noise_images, intermediate_shot)

	bg::Matrix{Float64} = 
		zeros(raw_img_size_x+2*half_padding_size, raw_img_size_y+2*half_padding_size)
	bg[half_padding_size+1:end-half_padding_size, 
	   	half_padding_size+1:end-half_padding_size] .= rand(raw_img_size_x, raw_img_size_y)
	intermediate_bg::Matrix{Float64} = 
		zeros(3*raw_img_size_x, 3*raw_img_size_y)
	apply_reflective_BC_bg!(bg, intermediate_bg)

	# Arrays to store intermediate variables
 	illuminated_object = zeros(Float64, sim_img_size_x+2*padding_size, sim_img_size_y+2*padding_size)
 	FFT_var = zeros(ComplexF64, sim_img_size_x+2*padding_size, sim_img_size_y+2*padding_size)
 	iFFT_var = zeros(ComplexF64, sim_img_size_x+2*padding_size, sim_img_size_y+2*padding_size)
 	img = zeros(ComplexF64, sim_img_size_x+2*padding_size, sim_img_size_y+2*padding_size)
 	img_abs = zeros(Float64, sim_img_size_x+2*padding_size, sim_img_size_y+2*padding_size)
 	mean_img = zeros(Float64, sim_img_size_x+2*padding_size, sim_img_size_y+2*padding_size)
 	mean_images = zeros(Float64, raw_img_size_x+2*half_padding_size, 
 			    raw_img_size_y+2*half_padding_size, n_patterns)
 	mod_fft_img = zeros(Float64, size(modulation_transfer_function_sim_vectorized)[1]) 


 
	@everywhere workers() begin
		n_accepted::Int64 = 0
		temperature::Float64 = 0.0

		sub_object::Matrix{Float64} = 
				zeros(sub_sim_img_size_x+2*padding_size, 
					  sub_sim_img_size_y+2*padding_size)
 		sub_shot_noise_imgs::Array{Float64} = 
				zeros(sub_raw_img_size_x+2*half_padding_size, 
				sub_raw_img_size_y+2*half_padding_size, 
				n_patterns)

 		sub_bg::Matrix{Float64} = 
				zeros(sub_raw_img_size_x+2*half_padding_size, 
				sub_raw_img_size_y+2*half_padding_size)


		mean_img = zeros(Float64, 2*half_padding_size+2, 2*half_padding_size+2)
 		mean_imgs_ij = zeros(Float64, 
							 2*quarter_padding_size+1, 
							 2*quarter_padding_size+1, 
							 n_patterns)
		proposed_mean_imgs_ij = zeros(Float64, 
									  2*quarter_padding_size+1, 
									  2*quarter_padding_size+1, 
									  n_patterns)
		illuminated_obj_ij = zeros(Float64, 2*padding_size+2, 2*padding_size+2)
		FFT_var = zeros(ComplexF64, 2*padding_size+2, 2*padding_size+2)
		iFFT_var = zeros(ComplexF64, 2*padding_size+2, 2*padding_size+2)
		img_ij = zeros(ComplexF64, 2*padding_size+2, 2*padding_size+2)
 		img_ij_abs = zeros(Float64, 2*padding_size+2, 2*padding_size+2)
 		mod_fft_img_ij = zeros(Float64, size(modulation_transfer_function_sim_ij_vectorized)[1]) 
		old_values = zeros(Float64, 2, 2) 
		proposed_values = zeros(Float64, 2, 2) 

	end

 
	# Variables to exchange data from other processors
	im_r::Vector{Int64} = zeros(n_procs_per_dim_x)
	ip_r::Vector{Int64} = zeros(n_procs_per_dim_x)
	jm_r::Vector{Int64} = zeros(n_procs_per_dim_y)
	jp_r::Vector{Int64} = zeros(n_procs_per_dim_y)

	im_s::Vector{Int64} = zeros(n_procs_per_dim_x)
	ip_s::Vector{Int64} = zeros(n_procs_per_dim_x)
	jm_s::Vector{Int64} = zeros(n_procs_per_dim_y)
	jp_s::Vector{Int64} = zeros(n_procs_per_dim_y)


	for i in 0:n_procs_per_dim_x-1

		im_r[i+1] = half_padding_size + i*sub_raw_img_size_x + 1 
		ip_r[i+1] = half_padding_size + (i+1)*sub_raw_img_size_x 

		im_s[i+1] = padding_size + i*sub_sim_img_size_x + 1 
		ip_s[i+1] = padding_size + (i+1)*sub_sim_img_size_x 

	end
	for j in 0:n_procs_per_dim_y-1

		jm_r[j+1] = half_padding_size + j*sub_raw_img_size_y + 1 
		jp_r[j+1] = half_padding_size + (j+1)*sub_raw_img_size_y 

		jm_s[j+1] = padding_size + j*sub_sim_img_size_y + 1 
		jp_s[j+1] = padding_size + (j+1)*sub_sim_img_size_y 

	end


	mcmc_log_posterior::Vector{Float64} = zeros(total_draws)
  	mcmc_log_posterior[draw] = compute_full_log_posterior!(object, 
  						shot_noise_images,
						bg,
  						illuminated_object,
  						FFT_var,
  						iFFT_var,
  						img,
  						img_abs,
  						mean_img,
  						mean_images,
  						mod_fft_img)
 

	n_accept::Int64 = 0
	temperature::Float64 = 0.0
	averaging_counter::Float64 = 0.0
	sub_img_sim::Matrix{Float64} = zeros(sub_sim_img_size_x+2*padding_size, sub_sim_img_size_y+2*padding_size)
	sub_img_shot::Array{Float64} = zeros(sub_raw_img_size_x+2*half_padding_size, 
					     sub_raw_img_size_y+2*half_padding_size,
					     n_patterns)
	sub_img_bg::Matrix{Float64} = zeros(sub_raw_img_size_x+2*half_padding_size, 
					     sub_raw_img_size_y+2*half_padding_size)



	for draw in 2:total_draws

		if draw > chain_burn_in_period
			temperature = 1.0 + (annealing_starting_temperature-1.0)*
					exp(-((draw-chain_burn_in_period-1) % 
					      annealing_frequency)/annealing_time_constant)
		elseif draw < chain_burn_in_period
       			temperature = 1.0 + (chain_starting_temperature-1.0)*
					exp(-((draw-1) % 
					      chain_burn_in_period)/chain_time_constant)
		end
		@show draw
		@show temperature
		flush(stdout);

		@everywhere workers() begin

  			temperature = $temperature
			draw = $draw
 
 			sub_object .= ($object)[im_sim:ip_sim, jm_sim:jp_sim]
  			sub_shot_noise_imgs .= ($shot_noise_images)[im_raw:ip_raw, jm_raw:jp_raw, :]
  			sub_bg .= ($bg)[im_raw:ip_raw, jm_raw:jp_raw]


			n_accepted = 0
			if i_procs + j_procs < draw - 1 
				if draw > chain_burn_in_period
					n_accepted = sample_object_neighborhood_MCMC!(temperature, 
							sub_object, 
							sub_shot_noise_imgs,
							sub_bg,
							illuminated_obj_ij,
							FFT_var,
							iFFT_var,
							img_ij,
 							img_ij_abs,
							mean_img,
							mean_imgs_ij,
							proposed_mean_imgs_ij,
							mod_fft_img_ij,
							old_values,
							proposed_values)

				else
					n_accepted = sample_object_neighborhood_MLE!(temperature, 
							sub_object, 
							sub_shot_noise_imgs,
							sub_bg,
							illuminated_obj_ij,
							FFT_var,
							iFFT_var,
							img_ij,
 							img_ij_abs,
							mean_img,
							mean_imgs_ij,
							proposed_mean_imgs_ij,
							mod_fft_img_ij,
							old_values,
							proposed_values)
				end
			end

		end

		n_accept = 0
 		for j in 1:n_procs_per_dim_y
  			for i in 1:n_procs_per_dim_x

 				procs_id::Int64 = ((j-1)*n_procs_per_dim_x+(i-1)+2)
     
 				sub_img_sim .= @fetchfrom procs_id sub_object
 				object[im_s[i]:ip_s[i], jm_s[j]:jp_s[j]] .= 
 						sub_img_sim[padding_size+1:end-padding_size,
       							padding_size+1:end-padding_size]
 
     
				sub_img_shot .= @fetchfrom procs_id sub_shot_noise_imgs	
 				shot_noise_images[im_r[i]:ip_r[i], jm_r[j]:jp_r[j], :] .= 
 						sub_img_shot[half_padding_size+1:end-half_padding_size,
       							half_padding_size+1:end-half_padding_size, :]
 
				sub_img_bg .= @fetchfrom procs_id sub_bg	
 				bg[im_r[i]:ip_r[i], jm_r[j]:jp_r[j]] .= 
 						sub_img_bg[half_padding_size+1:end-half_padding_size,
       							half_padding_size+1:end-half_padding_size]

     			accepted::Int64 = @fetchfrom procs_id n_accepted
     			n_accept += accepted
			end
  		end
		apply_reflective_BC_object!(object, intermediate_object)
		apply_reflective_BC_shot!(shot_noise_images, intermediate_shot)
		apply_reflective_BC_bg!(bg, intermediate_bg)


    
		println("accepted ", n_accept, " out of ", (2*raw_img_size_x * 2* raw_img_size_y) + (raw_img_size_x * raw_img_size_y), " pixels")	
		println("acceptance ratio = ", n_accept/ (2*raw_img_size_x * 2*raw_img_size_y + raw_img_size_x * raw_img_size_y))	
    		flush(stdout);

  		mcmc_log_posterior[draw] =
      				compute_full_log_posterior!(object, 
  							shot_noise_images,
 							bg,
  							illuminated_object,
  							FFT_var,
  							iFFT_var,
  							img,
  							img_abs,
  							mean_img,
  							mean_images,
  							mod_fft_img)

 		if (draw == chain_burn_in_period) || ((draw > chain_burn_in_period) &&
			((draw - chain_burn_in_period) % 
			 annealing_frequency > annealing_burn_in_period ||
			 (draw - chain_burn_in_period) % annealing_frequency == 0) &&
			((draw - chain_burn_in_period) % averaging_frequency == 0))
  
  			averaging_counter += 1.0
  			sum_object .+= object
  			mean_object .= sum_object ./ averaging_counter
  
  			println("Averaging Counter = ", averaging_counter)
  			println("Saving Data...")
  			flush(stdout);
 
 
			save_data(draw, 
					mcmc_log_posterior,
          			object, 
					shot_noise_images,
					bg,
          			mean_object,
          			averaging_counter)
		end

  		if plotting == true 
			if draw % plotting_frequency == 0 
    				plot_data(draw, object, mean_object, shot_noise_images, bg, mcmc_log_posterior)
			end
   		end
        
	end

	return nothing
end

@time sampler()
rmprocs(workers())
