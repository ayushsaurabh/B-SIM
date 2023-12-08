# B-SIM

###########################################################
# Copyright (C) 2022 Presse Lab - All Rights Reserved
#
# Author: Ayush Saurabh
#
# You may use, distribute and modify this code under the
# terms of the MIT license.
###########################################################


include("input_parameters.jl")

# These julia packages help simplify calculations and plotting.
using Distributed


println("Adding processors...")
flush(stdout);
addprocs(n_procs_per_dim^2, topology=:master_worker)
println("Done.")
flush(stdout);

@everywhere begin
	using Random, Distributions
	using TiffImages, FixedPointNumbers
	using LinearAlgebra
	using SpecialFunctions
	using FFTW
end

if plotting == true

	println("Plotting is On.")
	flush(stdout);

	using Plots
else

	println("Plotting is Off.")
	flush(stdout);

end
using HDF5

###############################################################################
# A brief description of the sampler:
#
# The sampler function below executes a Markov Chain Monte Carlo (MCMC)
# algorithm (Gibbs) where samples for each parameter of interest are generated
# sequentially from their corresponding probability distributions (posterior).
# First, the sampler creates/initiates arrays to store all the samples,
# posterior values, and acceptance rates for proposed samples. Next, new
# samples are then iteratively proposed using proposal (normal) distributions
# for each parameter, to be accepted or rejected by the Metropolis-Hastings
# step. If accepted, the proposed sample is stored in the arrays otherwise the
# previous sample is stored at the same MCMC iteration.
#
# The variance of the proposal distribution typically decides how often
# proposals are accepted/rejected. A larger covariance or movement away from
# the previous sample would lead to a larger change in likelihood/posterior
# values. Since the sampler prefers movement towards high probability regions,
# a larger movement towards low probability regions would lead to likely
# rejection of the sample compared to smaller movement.
#
# The collected samples can then be used to compute statistical quantities and
# plot probability distributions. The plotting function used by the sampler
# in this code allows monitoring of posterior values, most probable model for
# the molecule, and distributions over transition rates and FRET efficiencies.
#
# As mentioned before, sampler prefers movement towards higher probability
# regions of the posterior distribution. This means that if parameters are
# initialized in low probability regions of the posterior, which is typically
# the case, the posterior would appear to increase initially for many
# iterations (hundreds to thousands depending on the complexity of the model).
# This initial period is called burn-in. After burn-in, convergence is achieved
# where the posterior would typically fluctuate around some mean/average value.
# The convergence typically indicates that the sampler has reached the maximum
# of the posterior distribution (the most probability region), that is, sampler
# generates most samples from higher probability region. In other words, given
# a large collection of samples, the probability density in a region of
# parameter space is proportional to the number of samples collected from that
# region.
#
# All the samples collected during burn-in are usually ignored
# when computing statistical properties and presenting the final posterior
# distribution.
#
###############################################################################



@everywhere workers() include("input_parameters.jl")

function get_ground_truth()
	file_name = string(working_directory,
						"ground_truth_", illumination_file_name_suffix,".tif")
	ground_truth = TiffImages.load(file_name)
	ground_truth = Float64.(ground_truth)
	return ground_truth
end

println("Importing ground truth...")
flush(stdout);
const ground_truth = get_ground_truth()
println("Done.")
flush(stdout);


function get_camera_calibration_data()

	local offset_map::Matrix{Float64}
	local variance_map::Matrix{Float64}
	local error_map::Matrix{Float64}
	local gain_map::Matrix{Float64}


	if noise_maps_available == true
		file_name = string(working_directory,
							"offset_map.tif")
		offset_map = TiffImages.load(file_name)
		offset_map = Float64.(offset_map)

		file_name = string(working_directory,
							"variance_map.tif")
		variance_map = TiffImages.load(file_name)
		variance_map = Float64.(variance_map)

		file_name = string(working_directory,
							"gain_map.tif")
		gain_map = TiffImages.load(file_name)
		gain_map = Float64.(gain_map)
	else

		offset_map = offset .* ones(raw_image_size, raw_image_size)
		gain_map = gain .* ones(raw_image_size, raw_image_size)
		error_map = noise .* ones(raw_image_size, raw_image_size)

	end

	return offset_map, error_map, gain_map
end

println("Importing camera calibration data...")
flush(stdout);
const offset_map, error_map, gain_map = get_camera_calibration_data()
println("Done.")
println("approx gain = ", gain_map[1, 1])
println("approx offset = ", offset_map[1, 1])
println("approx error = ", error_map[1, 1])
flush(stdout);


function get_input_raw_images()
	file_name = string(working_directory,
						"raw_images_",raw_file_name_suffix, ".tif")
	raw_imgs = TiffImages.load(file_name)
	imgs = Matrix{Float64}[]
	for pattern in 1:9
		imgs = vcat(imgs, [Float64.(raw_imgs[:, :, pattern])])
	end
	return imgs
end

println("Importing raw images...")
flush(stdout);
const input_raw_images = get_input_raw_images()
println("Done.")
println("Size of Raw Images = ", size(input_raw_images[1]))
flush(stdout);

@everywhere const img_size = size($input_raw_images[1])[1]

function get_camera_calibration_data_with_ghosts(input_map::Matrix{Float64}, average_val)
	img::Matrix{Float64} = average_val .* ones(img_size+ghost_size, img_size+ghost_size)
	img[half_ghost_size+1:end-half_ghost_size,
						half_ghost_size+1:end-half_ghost_size] =
										input_map[1:end, 1:end]
	return img
end

println("Adding ghosts to calibration data...")
flush(stdout);

const gain_map_with_ghosts = get_camera_calibration_data_with_ghosts(gain_map, gain)
println("Size of gain map With ghosts = ", size(gain_map_with_ghosts))

const offset_map_with_ghosts = get_camera_calibration_data_with_ghosts(offset_map, offset)
println("Size of offset map With ghosts = ", size(offset_map_with_ghosts))

const error_map_with_ghosts = get_camera_calibration_data_with_ghosts(error_map, noise)
println("Size of error map With ghosts = ", size(error_map_with_ghosts))

println("Done.")
flush(stdout);



function get_raw_images_with_ghosts(input_raw_imgs::Vector{Matrix{Float64}})
	imgs = Matrix{Float64}[]
	for pattern in 1:9
		img = zeros(img_size+ghost_size, img_size+ghost_size)
		img[half_ghost_size+1:end-half_ghost_size,
						half_ghost_size+1:end-half_ghost_size] =
										input_raw_imgs[pattern][1:end, 1:end]
		imgs = vcat(imgs, [img])
	end
	return imgs
end

println("Adding ghosts to raw images...")
flush(stdout);
const raw_images_with_ghosts = get_raw_images_with_ghosts(input_raw_images)
println("Done.")
println("Size of Raw Images With Ghosts = ", size(raw_images_with_ghosts[1]))
flush(stdout);

function get_illumination_patterns()
	file_name = string(working_directory,
						"illumination_patterns_", illumination_file_name_suffix,".tif")

	illum_imgs = TiffImages.load(file_name)
	imgs = Matrix{Float64}[]
	for pattern in 1:9
		imgs = vcat(imgs, [Float64.(illum_imgs[:, :, pattern])])
	end
	return imgs
end


println("Importing illumination patterns...")
flush(stdout);
const illumination_patterns_without_ghosts = get_illumination_patterns()
println("Done.")
println("Size of Illumination Patterns = ",
				size(illumination_patterns_without_ghosts[1]))
flush(stdout);

function get_illumination_patterns_with_ghosts(
					input_illum_patterns::Vector{Matrix{Float64}})
	illum_patterns = Matrix{Float64}[]
	for pattern in 1:9
		illum_pattern = zeros(2*(img_size+ghost_size), 2*(img_size+ghost_size))
		illum_pattern[ghost_size+1:end-ghost_size,
						ghost_size+1:end-ghost_size] =
								input_illum_patterns[pattern][1:end, 1:end]
		illum_patterns = vcat(illum_patterns, [illum_pattern])
	end
	return illum_patterns
end

println("Adding ghosts to illumination patterns...")
flush(stdout);
const illumination_patterns = get_illumination_patterns_with_ghosts(
								illumination_patterns_without_ghosts)
println("Done.")
println("Size of Illumination Patterns With Ghosts = ",
				size(illumination_patterns[1]))
flush(stdout);


@everywhere workers() begin
	const i_procs::Integer = (myid()-2)%n_procs_per_dim
	const j_procs::Integer = (myid()-2 - i_procs)/n_procs_per_dim

	const im_raw::Integer = i_procs*img_size/n_procs_per_dim + 1
	const ip_raw::Integer = ghost_size + (i_procs+1)*img_size/n_procs_per_dim
	const jm_raw::Integer = j_procs*img_size/n_procs_per_dim + 1
	const jp_raw::Integer = ghost_size + (j_procs+1)*img_size/n_procs_per_dim
	const sub_size_raw::Integer = img_size/n_procs_per_dim+ghost_size

	const im_gt::Integer = i_procs*2*img_size/n_procs_per_dim + 1
	const ip_gt::Integer = 2*ghost_size + (i_procs+1)*2*img_size/n_procs_per_dim
	const jm_gt::Integer = j_procs*2*img_size/n_procs_per_dim + 1
	const jp_gt::Integer = 2*ghost_size + (j_procs+1)*2*img_size/n_procs_per_dim
	const sub_size_gt::Integer = 2*img_size/n_procs_per_dim+2*ghost_size
end

@everywhere workers() function get_sub_images(imgs::Vector{Matrix{Float64}},
				im::Integer, ip::Integer, jm::Integer, jp::Integer)
	sub_imgs = Matrix{Float64}[]
	for pattern in 1:9
		sub_imgs = vcat(sub_imgs, [imgs[pattern][im:ip, jm:jp]])
	end
	return sub_imgs
end

println("Assigning sections of raw images to each processor...")
flush(stdout);
@everywhere workers() const sub_raw_images = get_sub_images($raw_images_with_ghosts,
								im_raw, ip_raw, jm_raw, jp_raw)
println("Done.")
flush(stdout);

@everywhere workers() function get_sub_calibration_map(full_map::Matrix{Float64},
				im::Integer, ip::Integer, jm::Integer, jp::Integer)
	sub_map::Matrix{Float64} = full_map[im:ip, jm:jp]
	return sub_map
end

println("Assigning sections of calibration maps to each processor...")
flush(stdout);
@everywhere workers() const sub_gain_map = get_sub_calibration_map($gain_map_with_ghosts,
								im_raw, ip_raw, jm_raw, jp_raw)
@everywhere workers() const sub_offset_map = get_sub_calibration_map($offset_map_with_ghosts,
								im_raw, ip_raw, jm_raw, jp_raw)
@everywhere workers() const sub_error_map = get_sub_calibration_map($error_map_with_ghosts,
								im_raw, ip_raw, jm_raw, jp_raw)

println("Done.")
flush(stdout);

println("Assigning sections of illumination patterns to each processor...")
flush(stdout);
@everywhere workers() const sub_illumination_patterns =
					get_sub_images($illumination_patterns,
						im_gt, ip_gt, jm_gt, jp_gt)
println("Done.")
flush(stdout);

const grid_physical_1D = dx .* collect(-(img_size + ghost_size):
								(img_size + ghost_size - 1)) # in micrometers

@everywhere function incoherent_PSF(x_c::Vector{Float64}, x_e::Vector{Float64})
	return exp(-norm(x_c-x_e)^2/(2.0*sigma^2)) /
					(sqrt(2.0*pi) * sigma)^(size(x_e)[1])
end

println("Done.")
flush(stdout);

function FFT_incoherent_PSF()
	psf_on_grid = zeros(2*img_size+2*ghost_size, 2*img_size+2*ghost_size)
	for i in 1:2*img_size + 2*ghost_size
		for j in 1:2*img_size + 2*ghost_size
			x_e::Vector{Float64} = [grid_physical_1D[i], grid_physical_1D[j]]
			psf_on_grid[i, j] =  incoherent_PSF([0.0, 0.0], x_e)
		end
	end
	return fft(ifftshift(psf_on_grid))
end


println("Computing FFT of the PSF...")
flush(stdout);
const FFT_point_spread_function =  FFT_incoherent_PSF()
println("Done.")
flush(stdout);


function get_widefield_image(illumination_prof::Matrix{Float64},
										fluorophore_density::Matrix{Float64})
	illuminated_density::Matrix{Float64} =
					(illumination_prof .* fluorophore_density)
	FFT_illuminated_density::Matrix{ComplexF64} =
				fft(ifftshift(illuminated_density)) .* dx^2
	FFT_final::Matrix{ComplexF64} =
				FFT_point_spread_function .* FFT_illuminated_density

 	image::Matrix{Float64} = abs.(real.(fftshift(ifft(FFT_final))))

	return image
end

function get_mean_images(ground_truth::Matrix{Float64})
	mean_images = Matrix{Float64}[]
	for pattern in 1:9
		final_image::Matrix{Float64} =
			get_widefield_image(illumination_patterns[pattern], ground_truth)
		low_res_image::Matrix{Float64} = downsample_image(final_image, 2)
		mean_images = vcat(mean_images, [low_res_image])
	end
	return mean_images
end

@everywhere function downsample_image(input_image::Matrix{Float64},
									downsampling_factor::Integer)
	input_image_size::Integer = size(input_image)[1]
	downsampled_image_size::Integer = Integer(input_image_size/2)
	downsampled_image =
				zeros(downsampled_image_size, Integer(input_image_size/2))
	for i in 1:downsampled_image_size
		for j in 1:downsampled_image_size
			for l in 1:downsampling_factor
				for m in 1:downsampling_factor
					downsampled_image[i, j] +=
						input_image[downsampling_factor*(i-1)+l,
											downsampling_factor*(j-1)+m]
				end
			end
		end
	end
	return downsampled_image
end

################Inference Part#######################################
@everywhere workers() const grid_physical_1D_ij =
			dx .* collect(-(ghost_size+1):ghost_size) # in micrometers

@everywhere workers() function FFT_incoherent_PSF_ij()
	psf_on_grid = zeros(2*ghost_size+2, 2*ghost_size+2)
	for i in 1:2*ghost_size+2
		for j in 1:2*ghost_size+2
			x_e::Vector{Float64} = [grid_physical_1D_ij[i],
											grid_physical_1D_ij[j]]
			psf_on_grid[i, j] =  incoherent_PSF([0.0, 0.0], x_e)
		end
	end
	return fft(ifftshift(psf_on_grid))
end

println("Computing FFT of the PSF in neighborhood for inference...")
flush(stdout);
@everywhere workers() const FFT_point_spread_function_ij =
											FFT_incoherent_PSF_ij()
println("Done.")
flush(stdout);

@everywhere workers() function get_widefield_image_ij(
						illumination_prof::Matrix{Float64},
						fluorophore_density::Matrix{Float64},
						i::Integer, j::Integer)

  	i_minus::Integer = i-ghost_size
 	i_plus::Integer = i+ghost_size+1
 	j_minus::Integer = j-ghost_size
 	j_plus::Integer = j+ghost_size+1

	im::Integer = half_ghost_size + 1
	ip::Integer = ghost_size + 2 + half_ghost_size

	jm::Integer = half_ghost_size + 1
	jp::Integer = ghost_size + 2 + half_ghost_size



	illuminated_density::Matrix{Float64} = (illumination_prof[i_minus:i_plus,
											j_minus:j_plus] .*
											fluorophore_density[i_minus:i_plus,
											j_minus:j_plus])
	FFT_illuminated_density::Matrix{ComplexF64} =
					fft(ifftshift(illuminated_density)) .* dx^2
	FFT_final::Matrix{ComplexF64} =
				FFT_point_spread_function_ij .* FFT_illuminated_density

 	image::Matrix{Float64} = abs.(real.(fftshift(ifft(FFT_final))))

 	return image[im:ip, jm:jp]
end


@everywhere workers() function get_mean_images_ij(ground_truth::Matrix{Float64},
									i::Integer, j::Integer)
	mean_images_ij = Matrix{Float64}[]
	for pattern in 1:9
		final_image::Matrix{Float64} =
				get_widefield_image_ij(sub_illumination_patterns[pattern],
											ground_truth, i, j)
		low_res_image::Matrix{Float64} = downsample_image(final_image, 2)
		mean_images_ij = vcat(mean_images_ij, [low_res_image])
	end
	return mean_images_ij
end

@everywhere workers() function get_shot_noise_images_ij(ii::Integer, jj::Integer,
								shot_noise_images::Vector{Matrix{Float64}})

	i_minus::Integer = ii - half_ghost_size/2
	i_plus::Integer = ii + half_ghost_size/2
	j_minus::Integer = jj - half_ghost_size/2
	j_plus::Integer = jj + half_ghost_size/2

	shot_noise_imgs_ij = Matrix{Float64}[]
	for pattern in 1:9
 			shot_noise_imgs_ij = vcat(shot_noise_imgs_ij,
						[shot_noise_images[pattern][i_minus:i_plus,
													 	j_minus:j_plus]])
	end
	return shot_noise_imgs_ij
end


@everywhere workers() function get_log_likelihood_ij(
							ii::Integer, jj::Integer,
							mean_imgs_ij::Vector{Matrix{Float64}},
							shot_noise_imgs_ij::Vector{Matrix{Float64}})

	i_minus::Integer = 1
	i_plus::Integer = half_ghost_size+1

	if (i_procs== 0) && (0 < ii - half_ghost_size <=
						 					Integer(half_ghost_size/2))

		i_minus += (half_ghost_size/2 - (ii - half_ghost_size - 1))

	end
	if (i_procs==n_procs_per_dim-1) &&
			(0 <= (sub_size_raw - half_ghost_size) - ii <
			 								Integer(half_ghost_size/2))

		i_plus -= (half_ghost_size/2 + ii - (sub_size_raw - half_ghost_size))

	end

	j_minus::Integer = 1
	j_plus::Integer = half_ghost_size+1

	if (j_procs== 0) && (0 < jj - half_ghost_size <=
						 					Integer(half_ghost_size/2))

		j_minus += (half_ghost_size/2 - (jj - half_ghost_size - 1))

	end
	if (j_procs==n_procs_per_dim-1) &&
			(0 <= (sub_size_raw - half_ghost_size) - jj <
			 								Integer(half_ghost_size/2))

		j_plus -= (half_ghost_size/2 + jj - (sub_size_raw - half_ghost_size))

	end



	log_likelihood_ij::Float64 = 0.0
	for pattern in 1:9
		log_likelihood_ij += sum(logpdf.(
					Poisson.(mean_imgs_ij[pattern][i_minus:i_plus,
										j_minus:j_plus]),
						shot_noise_imgs_ij[pattern][i_minus:i_plus,
										j_minus:j_plus]))
	end

	return log_likelihood_ij
end


# The following Gibbs sampler computes likelihood for the neighborhood only.
@everywhere workers() function sample_gt_neighborhood(
									temperature::Float64,
									gt::Matrix{Float64},
									shot_noise_images::Vector{Matrix{Float64}})

	local ii::Integer
	local jj::Integer
	local shot_noise_imgs_ij::Vector{Matrix{Float64}}
	local mean_imgs_ij::Vector{Matrix{Float64}}
	local old_log_likelihood::Float64
	local old_log_prior::Float64
	local old_jac::Float64
	local old_log_posterior::Float64
	local proposed_mean_imgs_ij::Vector{Matrix{Float64}}
	local new_log_likelihood::Float64
	local new_log_prior::Float64
	local new_jac::Float64
	local new_log_posterior::Float64
	local log_hastings::Float64
	local expected_photon_count::Float64
	local proposed_shot_noise_pixel::Float64
	local log_forward_proposal_probability::Float64
	local log_backward_proposal_probability::Float64


	for i in collect(ghost_size+1:2:sub_size_gt-ghost_size)
		for j in collect(ghost_size+1:2:sub_size_gt-ghost_size)

 			ii = ceil(i/2)
 			jj = ceil(j/2)

			shot_noise_imgs_ij = get_shot_noise_images_ij(ii, jj,
										shot_noise_images)

			mean_imgs_ij = get_mean_images_ij(gt, i, j)

			old_log_likelihood = get_log_likelihood_ij(ii, jj,
									mean_imgs_ij, shot_noise_imgs_ij)

			old_log_prior = 0.0
			old_jac = 0.0
			for l in 0:1
				for m in 0:1
					old_log_prior += logpdf(Gamma(gamma_prior_shape, gamma_prior_scale), gt[i+l, j+m])
					old_jac += log(gt[i+l, j+m])
				end
			end
			old_log_posterior = old_log_likelihood + old_log_prior

			proposed_gt = copy(gt)
			for l in 0:1
				for m in 0:1
					proposed_gt[i+l, j+m] =
						rand(Normal(log(gt[i+l, j+m]),
										covariance_gt), 1)[1]
					proposed_gt[i+l, j+m] = exp.(proposed_gt[i+l, j+m])
				end
			end
			proposed_mean_imgs_ij = get_mean_images_ij(proposed_gt, i, j)

			new_log_likelihood = get_log_likelihood_ij(ii, jj,
							proposed_mean_imgs_ij, shot_noise_imgs_ij)

			new_log_prior = 0.0
			new_jac = 0.0
			for l in 0:1
				for m in 0:1
					new_log_prior += logpdf(Gamma(gamma_prior_shape, gamma_prior_scale),
												proposed_gt[i+l, j+m])
					new_jac += log(proposed_gt[i+l, j+m])
				end
			end
			new_log_posterior = new_log_likelihood + new_log_prior

			log_hastings = (1.0/temperature) *
                        (new_log_posterior - old_log_posterior) +
									new_jac - old_jac



			if log_hastings > log(rand())
				gt = copy(proposed_gt)
				mean_imgs_ij = copy(proposed_mean_imgs_ij)
			end



			# Sample Intermediate Expected Photon Counts on each Pixel
  			old_log_likelihood = 0.0
  			for pattern in 1:9

				# Choose the central pixel in the mean images
				# for expected photon count
				expected_photon_count = mean_imgs_ij[pattern][
								Integer(half_ghost_size/2+1),
								Integer(half_ghost_size/2+1)]

  				old_log_likelihood = logpdf(Normal(sub_gain_map[ii, jj]*
                           		shot_noise_images[pattern][ii, jj] +
  								sub_offset_map[ii, jj],
                              	sub_error_map[ii, jj]),
 								sub_raw_images[pattern][ii, jj])

  				old_log_prior = logpdf(Poisson(expected_photon_count),
  								shot_noise_images[pattern][ii, jj])

   				old_log_posterior = old_log_likelihood  + old_log_prior

  				proposed_shot_noise_pixel =
 					rand(Poisson(shot_noise_images[pattern][ii, jj]), 1)[1]

  				new_log_likelihood = logpdf(Normal(sub_gain_map[ii, jj]*
                              	proposed_shot_noise_pixel +
  							  	sub_offset_map[ii, jj],
 								sub_error_map[ii, jj]),
 								sub_raw_images[pattern][ii, jj])

  				new_log_prior = logpdf(Poisson(expected_photon_count),
  								proposed_shot_noise_pixel)

    			new_log_posterior = new_log_likelihood  + new_log_prior

				log_forward_proposal_probability =
						logpdf(Poisson(shot_noise_images[pattern][ii, jj]),
  									proposed_shot_noise_pixel)

				log_backward_proposal_probability =
						logpdf(Poisson(proposed_shot_noise_pixel),
  									shot_noise_images[pattern][ii, jj])

  				log_hastings = (1.0/temperature)*
                              	(new_log_posterior - old_log_posterior) +
								log_backward_proposal_probability -
								log_forward_proposal_probability

  				if log_hastings > log(rand())
  					shot_noise_images[pattern][ii, jj] =
                                     proposed_shot_noise_pixel
  				end
  			end

		end
	end
	return gt, shot_noise_images
end

function get_log_likelihood(ground_truth::Matrix{Float64},
							shot_noise_images::Vector{Matrix{Float64}})

	log_likelihood::Float64 = 0.0

	mean_images::Vector{Matrix{Float64}} = get_mean_images(ground_truth)
	val_range = collect(half_ghost_size+1:1:half_ghost_size+img_size)
	for pattern in 1:9
		log_likelihood += sum(logpdf.(Poisson.(
				mean_images[pattern][val_range, val_range]),
				shot_noise_images[pattern][ val_range, val_range ]))
		log_likelihood += sum(logpdf.(Normal.(
                (gain_map_with_ghosts[val_range, val_range] .*
				shot_noise_images[pattern][ val_range, val_range]) .+
					offset_map_with_ghosts[val_range, val_range],
                    error_map_with_ghosts[val_range, val_range]),
					raw_images_with_ghosts[pattern][val_range, val_range]))
	end

	return log_likelihood
end

function compute_full_log_posterior(gt::Matrix{Float64},
							shot_noise_images::Vector{Matrix{Float64}})

	log_likelihood::Float64 = get_log_likelihood(gt, shot_noise_images)
	log_prior::Float64 = sum(logpdf.(Gamma(gamma_prior_shape, gamma_prior_scale),
							(gt[ghost_size+1:end-ghost_size,
								ghost_size+1:end-ghost_size].+eps(1.0))))
	log_posterior::Float64 = log_likelihood + log_prior

	@show log_likelihood, log_prior, log_posterior
	return log_posterior
end

function sample_gt(draw::Integer, gt::Matrix{Float64},
			shot_noise_imgs::Vector{Matrix{Float64}})

	if draw > initial_burn_in_period
		temperature = 1.0 + (annealing_starting_temperature-1.0)*
				exp(-((draw-1) % annealing_frequency)/annealing_time_constant)
	else
		temperature = 1.0
	end
	println("Temperature = ", temperature)
	flush(stdout);


	@everywhere workers() begin

		temperature = $temperature

		sub_gt::Matrix{Float64} = ($gt)[im_gt:ip_gt, jm_gt:jp_gt]
		sub_shot_noise_imgs::Vector{Matrix{Float64}} =
						get_sub_images($shot_noise_imgs,
								im_raw, ip_raw, jm_raw, jp_raw)

		sub_gt, sub_shot_noise_imgs =
				sample_gt_neighborhood(temperature, sub_gt, sub_shot_noise_imgs)

	end


 	local im::Integer
 	local ip::Integer
 	local jm::Integer
 	local jp::Integer
 	local sub_imgs::Vector{Matrix{Float64}}

 	for i in 0:n_procs_per_dim-1
 		for j in 0:n_procs_per_dim-1

 			im = ghost_size+i*2*img_size/n_procs_per_dim + 1
 			ip = ghost_size + (i+1)*2*img_size/n_procs_per_dim
 			jm = ghost_size+j*2*img_size/n_procs_per_dim + 1
 			jp = ghost_size + (j+1)*2*img_size/n_procs_per_dim

 			gt[im:ip, jm:jp] = @fetchfrom (j*n_procs_per_dim+i+2) sub_gt[
 							ghost_size+1:end-ghost_size,
 							ghost_size+1:end-ghost_size]

 			im = half_ghost_size + i*img_size/n_procs_per_dim + 1
 			ip = half_ghost_size + (i+1)*img_size/n_procs_per_dim
 			jm = half_ghost_size + j*img_size/n_procs_per_dim + 1
 			jp = half_ghost_size + (j+1)*img_size/n_procs_per_dim

 			sub_imgs =@fetchfrom (j*n_procs_per_dim+i+2) sub_shot_noise_imgs
 			for pattern in 1:9
 				shot_noise_imgs[pattern][im:ip, jm:jp] =
 						 sub_imgs[pattern][
 							half_ghost_size+1:end-half_ghost_size,
 							half_ghost_size+1:end-half_ghost_size]
 			end
 		end
 	end

	return gt, shot_noise_imgs
end

function save_data(current_draw::Integer,
					mcmc_log_posterior::Vector{Float64},
					gt::Matrix{Float64},
					shot_noise_images::Vector{Matrix{Float64}},
					MAP_index::Integer,
					gt_MAP::Matrix{Float64},
					gt_mean::Matrix{Float64},
					gt_variance::Matrix{Float64},
					averaging_counter::Float64)

	# Save the data in HDF5 format.
	file_name = string(working_directory, "mcmc_output_", illumination_file_name_suffix,
									"_", current_draw, ".h5")

	fid = h5open(file_name,"w")
	write_dataset(fid, string("averaging_counter"), averaging_counter)
	write_dataset(fid, string("inferred_density"), gt)
	write_dataset(fid, string("MAP_index"), MAP_index)
	write_dataset(fid, string("MAP_inferred_density"), gt_MAP)
	write_dataset(fid, string("mean_inferred_density"), gt_mean)
	write_dataset(fid, string("variance_inferred_density"), gt_variance)

	for pattern in 1:9
			write_dataset(fid, string("shot_noise_images_", pattern),
										shot_noise_images[pattern])
	end
	write_dataset(fid, "mcmc_log_posteriors",
								mcmc_log_posterior[1:current_draw])
	close(fid)

	return nothing
end

if plotting == true
	function plot_results(draw, mcmc_log_posterior, gt, shot_noise_images, mean_gt)

		if draw > posterior_moving_window_size
			plot_a = plot(collect(draw-posterior_moving_window_size:draw), 
						mcmc_log_posterior[draw-posterior_moving_window_size:draw], 
						size=(2000, 2000),
						legend=false,
						title = "log-Posterior",
						xlabel = "Iterations");
		else
			plot_a = plot(collect(1:draw),
						mcmc_log_posterior[1:draw], 
						size=(2000, 2000),
						legend=false,
						title = "log-Posterior",
						xlabel = "Iterations");
		end


		plot_b = heatmap(gt[ghost_size+1:end-ghost_size,
					ghost_size+1:end-ghost_size],
					c=:grays, legend=false, size=(2000, 2000),
					title = "Current SIM Sample");

		plot_c = heatmap(input_raw_images[1],
					c=:grays, legend=false, size=(2000, 2000),
				 	title = "A Raw Image");
		plot_d = heatmap(shot_noise_images[1][
					half_ghost_size+1:end-half_ghost_size,
					half_ghost_size+1:end-half_ghost_size],
					c=:grays, legend=false, size=(2000, 2000),
					title = "Shot Noise Image");

		plot_e = heatmap(ground_truth,
					c=:grays, legend=false, size=(2000, 2000),
					title = "Ground Truth");
		plot_f = heatmap(mean_gt[ghost_size+1:end-ghost_size,
					ghost_size+1:end-ghost_size],
					c=:grays, legend=false, size=(2000, 2000),
					title = "Mean SIM Image");

		display(plot(plot_c, plot_d, plot_e, plot_a, plot_b, plot_f,
						 		layout = (2, 3), size = (3000, 2000)))
		return nothing
	end
end

function sampler_SIM(draws::Integer, initial_inferred_density::Matrix{Float64},
						initial_shot_noise_images::Vector{Matrix{Float64}})

	# Initialize
	draw::Integer = 1
	println("draw = ", draw)
	flush(stdout);

  	gt::Matrix{Float64} = copy(initial_inferred_density)
	MAP_index::Integer = 1
   	MAP_gt::Matrix{Float64} = copy(gt)
   	sum_gt::Matrix{Float64} =
 				zeros(2*img_size+2*ghost_size, 2*img_size+2*ghost_size)
   	mean_gt::Matrix{Float64} =
 				zeros(2*img_size+2*ghost_size, 2*img_size+2*ghost_size)
   	sum_squared_gt::Matrix{Float64} =
 				zeros(2*img_size+2*ghost_size, 2*img_size+2*ghost_size)
   	mean_squared_gt::Matrix{Float64} =
 				zeros(2*img_size+2*ghost_size, 2*img_size+2*ghost_size)
   	variance_gt::Matrix{Float64} =
 				zeros(2*img_size+2*ghost_size, 2*img_size+2*ghost_size)



	shot_noise_images::Vector{Matrix{Float64}} = copy(initial_shot_noise_images)

	mcmc_log_posterior::Vector{Float64} = zeros(draws)
  	mcmc_log_posterior[draw] =
  				compute_full_log_posterior(gt, shot_noise_images)

	averaging_counter::Float64 = 0.0

	if plotting == true
		plot_results(draw, mcmc_log_posterior, gt, shot_noise_images, mean_gt)
	end

	for draw in 2:draws

		println("draw = ", draw)
		flush(stdout);

		gt, shot_noise_images =
					sample_gt(draw, gt, shot_noise_images)
  		mcmc_log_posterior[draw] =
  					compute_full_log_posterior(gt, shot_noise_images)

		if mcmc_log_posterior[draw] == maximum(mcmc_log_posterior[1:draw])
			MAP_index = draw
			MAP_gt = copy(gt)
		end

		if (draw >= initial_burn_in_period) &&
   				(draw % annealing_frequency >= annealing_burn_in_period) &&
					(draw % averaging_frequency == 0)

 			averaging_counter += 1.0
 			sum_gt += copy(gt)
 			sum_squared_gt += copy(gt).^2
 			mean_gt = sum_gt ./ averaging_counter
 			mean_squared_gt = sum_squared_gt ./ averaging_counter
 			variance_gt = mean_squared_gt .- (mean_gt.^2)

 			println("Averaging Counter = ", averaging_counter)
 			println("Saving Data...")
 			flush(stdout);

 			save_data(draw, mcmc_log_posterior,
 					gt, shot_noise_images,
					MAP_index,
					MAP_gt,
 					mean_gt, variance_gt,
 					averaging_counter)

 			println("Done.")
 			flush(stdout);

 		end

		if draw % plotting_frequency == 0 && plotting == true
			plot_results(draw, mcmc_log_posterior,
						gt, shot_noise_images, mean_gt)
		end

		# Garbage Collection and free memory
 		@everywhere GC.gc()
	end

	return gt, shot_noise_images
end


println("Initializing SIM Reconstruction...")
flush(stdout);

# Initialize inferred images
inferred_density = zeros(2*img_size+2*ghost_size, 2*img_size+2*ghost_size)
inferred_density[ghost_size+1:end-ghost_size,
			ghost_size+1:end-ghost_size]=rand(2*img_size, 2*img_size)
inferred_shot_noise_images = deepcopy(raw_images_with_ghosts)

for pattern in 1:9
		inferred_shot_noise_images[pattern][half_ghost_size+1:end-half_ghost_size,
											half_ghost_size+1:end-half_ghost_size] =
			round.(abs.((input_raw_images[pattern] .- offset) ./ gain))
end

println("Starting sampler...")
flush(stdout);

inferred_density, inferred_shot_noise_images =
		sampler_SIM(total_draws, inferred_density, inferred_shot_noise_images)

# Kill all the processes
rmprocs(workers())
