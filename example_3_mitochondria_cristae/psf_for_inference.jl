grid_physical_1D_ij = dx .* collect(-(padding_size+1):padding_size) # in micrometers
grid_physical_length_ij = (2.0*padding_size+1)*dx			

df_ij = 1/(grid_physical_length_ij) # Physcially correct spacing in spatial frequency space in units of micrometer^-1 

f_corrected_grid_1D_ij = df_ij .* collect(-(padding_size+1):padding_size)

mtf_on_grid_ij = zeros(2*padding_size+2, 2*padding_size+2)
psf_on_grid_ij = zeros(2*padding_size+2, 2*padding_size+2)

if psf_type == "airy_disk"
	function incoherent_PSF(x_c::Vector{Float64}, x_e::Vector{Float64})
		f_number::Float64 = 1/(2*numerical_aperture) ##approx
		return (jinc(norm(x_c - x_e)/(light_wavelength*f_number)))^2
	end
	for j in 1:2*padding_size+2
		for i in 1:2*padding_size+2
			x_e::Vector{Float64} = [grid_physical_1D_ij[i],
							grid_physical_1D_ij[j]]
			psf_on_grid_ij[i, j] =  incoherent_PSF([0.0, 0.0], x_e)
		end
	end

	normalization = sum(psf_on_grid_ij) * dx^2
	psf_on_grid_ij = psf_on_grid_ij ./ normalization
	psf_on_grid_ij = psf_on_grid_ij ./ sum(psf_on_grid_ij)


	intermediate_img = fftshift(fft(ifftshift(psf_on_grid_ij)))

	function MTF(f_vector::Vector{Float64})
		f_number::Float64 = 1/(2*numerical_aperture) ##approx
		highest_transmitted_frequency = 1.0 / (light_wavelength*f_number) 
		norm_f = norm(f_vector) / highest_transmitted_frequency
		if norm_f < 1.0
			mtf = 2.0/pi * (acos(norm_f) - norm_f*sqrt(1 - norm_f^2))
		elseif norm_f > 1.0
			mtf = 0.0
		end
		return mtf
	end
	for j in 1:2*padding_size+2
		for i in 1:2*padding_size+2
				mtf_on_grid_ij[i, j] = MTF([f_corrected_grid_1D_ij[i], f_corrected_grid_1D_ij[j]]) 
				if mtf_on_grid_ij[i, j] == 0.0
					intermediate_img[i, j] = 0.0 * intermediate_img[i, j]
				end
		end
	end
	const FFT_point_spread_function_ij::Matrix{ComplexF64} = 
							ifftshift(intermediate_img) .* dx^2


	function incoherent_PSF_SIM(x_c::Vector{Float64}, x_e::Vector{Float64})
		f_number::Float64 = 0.5 * 1/(2*numerical_aperture) ##approx
		return (jinc(norm(x_c - x_e)/(light_wavelength*f_number)))^2
	end
	for j in 1:2*padding_size+2
		for i in 1:2*padding_size+2
			x_e::Vector{Float64} = [grid_physical_1D_ij[i],
							grid_physical_1D_ij[j]]
			psf_on_grid_ij[i, j] =  incoherent_PSF_SIM([0.0, 0.0], x_e)
		end
	end

	normalization = sum(psf_on_grid_ij) * dx^2
	psf_on_grid_ij = psf_on_grid_ij ./ normalization
	psf_on_grid_ij = psf_on_grid_ij ./ sum(psf_on_grid_ij)

	intermediate_img = fftshift(fft(ifftshift(psf_on_grid_ij)))

	function MTF_SIM(f_vector::Vector{Float64})
		f_number::Float64 = 0.5 * 1/(2*numerical_aperture) ##approx
		highest_transmitted_frequency = 1.0 / (light_wavelength*f_number) 
		norm_f = norm(f_vector) / highest_transmitted_frequency
		if norm_f < 1.0
			mtf = 2.0/pi * (acos(norm_f) - norm_f*sqrt(1 - norm_f^2))
		elseif norm_f > 1.0
			mtf = 0.0
		end
		return mtf
	end
	for j in 1:2*padding_size+2
		for i in 1:2*padding_size+2
				mtf_on_grid_ij[i, j] = MTF_SIM([f_corrected_grid_1D_ij[i], f_corrected_grid_1D_ij[j]]) 
				if mtf_on_grid_ij[i, j] == 0.0
					intermediate_img[i, j] = 0.0 * intermediate_img[i, j]
				end
		end
	end
	const FFT_point_spread_function_sim_ij::Matrix{ComplexF64} = 
							ifftshift(intermediate_img) .* dx^2

elseif psf_type == "gaussian"
	function incoherent_PSF(x_c::Vector{Float64}, x_e::Vector{Float64})
		return exp(-norm(x_c-x_e)^2/(2.0*sigma^2)) /
					(sqrt(2.0*pi) * sigma)^(size(x_e)[1])
	end
	for j in 1:2*padding_size+2
		for i in 1:2*padding_size+2
			x_e::Vector{Float64} = [grid_physical_1D_ij[i],
							grid_physical_1D_ij[j]]
			psf_on_grid_ij[i, j] =  incoherent_PSF([0.0, 0.0], x_e)
		end
	end
	psf_on_grid_ij = psf_on_grid_ij ./ sum(psf_on_grid_ij)

	const FFT_point_spread_function_ij::Matrix{ComplexF64} = 
		fft(ifftshift(psf_on_grid_ij)) .* dx^2

	function incoherent_PSF_SIM(x_c::Vector{Float64}, x_e::Vector{Float64})
		sigma_sim::Float64 = sigma/2.0
		return exp(-norm(x_c-x_e)^2/(2.0*sigma_sim^2)) /
					(sqrt(2.0*pi) * sigma_sim)^(size(x_e)[1])
	end
	for j in 1:2*padding_size+2
		for i in 1:2*padding_size+2
			x_e::Vector{Float64} = [grid_physical_1D_ij[i],
							grid_physical_1D_ij[j]]
			psf_on_grid_ij[i, j] =  incoherent_PSF_SIM([0.0, 0.0], x_e)
		end
	end
	psf_on_grid_ij = psf_on_grid_ij ./ sum(psf_on_grid_ij)

	const FFT_point_spread_function_sim_ij::Matrix{ComplexF64} = 
		fft(ifftshift(psf_on_grid_ij)) .* dx^2

end

const fft_plan = plan_fft(ComplexF64.(psf_on_grid_ij))
const ifft_plan = plan_ifft(ComplexF64.(psf_on_grid_ij))

const modulation_transfer_function_ij::Matrix{Float64} = 
		abs.(fftshift(FFT_point_spread_function_ij)) 
const modulation_transfer_function_ij_vectorized::Vector{Float64} = 
		vec(modulation_transfer_function_ij) ./ sum(modulation_transfer_function_ij)

const modulation_transfer_function_sim_ij::Matrix{Float64} = 
		abs.(fftshift(FFT_point_spread_function_sim_ij)) 
const modulation_transfer_function_sim_ij_vectorized::Vector{Float64} = 
		vec(modulation_transfer_function_sim_ij) ./ sum(modulation_transfer_function_sim_ij)

grid_physical_ij = 0
normalization = 0
psf_on_grid = 0
mtf_on_grid = 0
intermediate_img = 0
f_corrected_grid_1D_ij

GC.gc()
