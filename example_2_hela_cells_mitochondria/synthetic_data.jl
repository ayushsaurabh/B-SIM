using Random, Distributions
using LinearAlgebra
using SpecialFunctions
using FFTW
using HDF5
using TiffImages
using Plots

include("input_parameters.jl")
file_name = string(working_directory, "mean_inferred_object_10.0.tif")
#				"ground_truth.tif")
ground_truth = TiffImages.load(file_name)
ground_truth = Float64.(ground_truth)

heatmap(ground_truth)

img_size_x = size(ground_truth)[1]
img_size_y = size(ground_truth)[2]


function add_padding(input_raw_img::Matrix{Float64})
	img = Matrix{Float64}[]
	img = zeros(img_size_x+2*padding_size, img_size_y+2*padding_size)
	img[padding_size+1:end-padding_size,
				padding_size+1:end-padding_size] =
					input_raw_img[:, :]
	return img
end

ground_truth_with_padding = add_padding(ground_truth)


grid_physical_1D_x = dx .* collect(-(img_size_x/2 + padding_size):
					(img_size_x/2 + padding_size - 1)) # in micrometers
grid_physical_1D_y = dx .* collect(-(img_size_y/2 + padding_size):
					(img_size_y/2 + padding_size - 1)) # in micrometers

grid_physical_length_x = (img_size_x + 2*padding_size - 1)*dx			
grid_physical_length_y = (img_size_y + 2*padding_size - 1)*dx			
                    
                    
df_x = 1/(grid_physical_length_x) # Physcially correct spacing in spatial frequency space in units of micrometer^-1 
df_y = 1/(grid_physical_length_y) # Physcially correct spacing in spatial frequency space in units of micrometer^-1 
                    
f_corrected_grid_1D_x = df_x .* collect(-(img_size_x/2 + padding_size):
            (img_size_x/2 + padding_size - 1)) # in units of micrometer^-1
f_corrected_grid_1D_y = df_y .* collect(-(img_size_y/2 + padding_size):
            (img_size_y/2 + padding_size - 1)) # in units of micrometer^-1
                    
mtf_on_grid = zeros(img_size_x+2*padding_size, img_size_y+2*padding_size)
psf_on_grid = zeros(img_size_x+2*padding_size, img_size_y+2*padding_size)
                    
function incoherent_PSF(x_c::Vector{Float64}, x_e::Vector{Float64})
    f_number::Float64 = 1/(2*numerical_aperture) ##approx
    return (jinc(norm(x_c - x_e)/(light_wavelength*f_number)))^2
end
                     
for j in 1:img_size_y + 2*padding_size
    for i in 1:img_size_x + 2*padding_size
        x_e::Vector{Float64} = [grid_physical_1D_x[i], grid_physical_1D_y[j]]
        psf_on_grid[i, j] =  incoherent_PSF([0.0, 0.0], x_e)
    end
end
normalization = sum(psf_on_grid) * dx^2
psf_on_grid = psf_on_grid ./ normalization
intermediate_img = fftshift(fft(ifftshift(psf_on_grid)))
                    
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
                    
for j in 1:img_size_y + 2*padding_size
    for i in 1:img_size_x + 2*padding_size
            mtf_on_grid[i, j] = MTF([f_corrected_grid_1D_x[i], f_corrected_grid_1D_y[j]]) 
            if mtf_on_grid[i, j] == 0.0
                intermediate_img[i, j] = 0.0 * intermediate_img[i, j]
            end
    end
end
const FFT_point_spread_function::Matrix{ComplexF64} = ifftshift(intermediate_img) 

heatmap(fftshift(abs.(FFT_point_spread_function)) .^0.3)

function airy_disk_incoherent_psf(x_c, x_e)
    f_number = 1/(2*numerical_aperture) ##approx
    return (jinc(norm(x_c - x_e)/(light_wavelength*f_number)))^2
end

function FFT_incoherent_airy_disk_PSF()
	psf_on_grid = zeros(img_size_x+2*padding_size, img_size_y+2*padding_size)
	for i in 1:img_size_x + 2*padding_size
		for j in 1:img_size_y + 2*padding_size
			x_e::Vector{Float64} = [grid_physical_1D_x[i], grid_physical_1D_y[j]]
			psf_on_grid[i, j] =  airy_disk_incoherent_psf([0.0, 0.0], x_e)
		end
	end
    normalization = sum(psf_on_grid) * dx^2
    psf_on_grid = psf_on_grid ./ normalization
	return fft(ifftshift(psf_on_grid))
end

FFT_point_spread_function =  FFT_incoherent_airy_disk_PSF()




function gaussian_incoherent_PSF(x_c::Vector{Float64}, x_e::Vector{Float64})
	return exp(-norm(x_c-x_e)^2/(2.0*sigma^2)) /
					(sqrt(2.0*pi) * sigma)^(size(x_e)[1])
end

function FFT_incoherent_gaussian_PSF()
	psf_on_grid = zeros(img_size_x+2*padding_size, img_size_y+2*padding_size)
	for i in 1:img_size_x + 2*padding_size
		for j in 1:img_size_y + 2*padding_size
			x_e::Vector{Float64} = [grid_physical_1D_x[i], grid_physical_1D_y[j]]
			psf_on_grid[i, j] =  gaussian_incoherent_PSF([0.0, 0.0], x_e)
		end
	end
	return fft(ifftshift(psf_on_grid))
end

FFT_point_spread_function =  FFT_incoherent_gaussian_PSF()



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

flatfield = 0.01 .* ones(img_size_x, img_size_y)
flatfield = add_padding(flatfield)
mean_image = get_widefield_image(flatfield,
                            ground_truth_with_padding)

heatmap(ground_truth_with_padding)                      
heatmap(mean_image)

poisson_image = rand.(Poisson.(mean_image))
heatmap(poisson_image)

camera_image = rand.(Normal.(gain .* poisson_image .+ offset, noise))
heatmap(camera_image)

heatmap((abs.(fftshift(fft(ifftshift(camera_image)))) ).^ 0.1)

grays = convert.(Gray{Float64}, flatfield[
						padding_size+1:end-padding_size,
                             padding_size+1:end-padding_size])
img = TiffImages.DenseTaggedImage(grays)
TiffImages.save(string(working_directory,
		"flatfield",".tif"), img)

grays = convert.(Gray{Float64}, camera_image[
						padding_size+1:end-padding_size,
                             padding_size+1:end-padding_size])
img = TiffImages.DenseTaggedImage(grays)
TiffImages.save(string(working_directory,
		"raw_image_test_gaussian",".tif"), img)

grays = convert.(Gray{Float64}, poisson_image[
						padding_size+1:end-padding_size,
                             padding_size+1:end-padding_size])
img = TiffImages.DenseTaggedImage(grays)
TiffImages.save(string(working_directory,
		"poisson_image_test_gaussian",".tif"), img)

grays = convert.(Gray{Float64}, mean_image[
						padding_size+1:end-padding_size,
                             padding_size+1:end-padding_size])
img = TiffImages.DenseTaggedImage(grays)
TiffImages.save(string(working_directory,
		"mean_image_test_gaussian",".tif"), img)



modulation_transfer_function = abs.(fftshift(FFT_point_spread_function))[padding_size+1:end-padding_size, padding_size+1:end-padding_size]
modulation_transfer_function_vectorized = vec(modulation_transfer_function) ./ sum(modulation_transfer_function)

fft_image = vec(abs.(fftshift(fft(ifftshift(ground_truth))))) .+ eps()
logpdf(Dirichlet(modulation_transfer_function_vectorized),
                                fft_image ./ sum(fft_image))

fft_image = vec(abs.(fftshift(fft(ifftshift(camera_image[
						padding_size+1:end-padding_size,
                             padding_size+1:end-padding_size]))))) .+ eps()
logpdf(Dirichlet(modulation_transfer_function_vectorized),
                                fft_image ./ sum(fft_image))

fft_image = vec(abs.(fftshift(fft(ifftshift(poisson_image[
						padding_size+1:end-padding_size,
                             padding_size+1:end-padding_size]))))) .+ eps()
logpdf(Dirichlet(modulation_transfer_function_vectorized),
                                fft_image ./ sum(fft_image))

fft_image = vec(abs.(fftshift(fft(ifftshift(mean_image[
						padding_size+1:end-padding_size,
                             padding_size+1:end-padding_size]))))) .+eps()
plot(fft_image)
logpdf(Dirichlet(modulation_transfer_function_vectorized),
                                fft_image ./ sum(fft_image))

file_name = string(working_directory,
				"raw_image.tif")
raw_image = TiffImages.load(file_name)
raw_image = Float64.(raw_image)

file_name = string(working_directory,
				"ground_truth.tif")
grnd_truth = TiffImages.load(file_name)
grnd_truth = Float64.(grnd_truth)

file_name = string(working_directory,
				"poisson_image.tif")
grnd_truth_poisson = TiffImages.load(file_name)
grnd_truth_poisson = Float64.(grnd_truth_poisson)

file_name = string(working_directory,
				"flatfield.tif")
flatfield = TiffImages.load(file_name)
flatfield = Float64.(flatfield)


heatmap(grnd_truth)


fft_image = vec(abs.(fftshift(fft(ifftshift(grnd_truth))))) .+ eps()
logpdf(Dirichlet(modulation_transfer_function_vectorized),
                                fft_image ./ sum(fft_image))

fft_image = vec(abs.(fftshift(fft(ifftshift(raw_image))))) .+ eps()
logpdf(Dirichlet(modulation_transfer_function_vectorized),
                                fft_image ./ sum(fft_image))

fft_image = vec(abs.(fftshift(fft(ifftshift(grnd_truth_poisson))))) .+ eps()
logpdf(Dirichlet(modulation_transfer_function_vectorized),
                                fft_image ./ sum(fft_image))

ground_truth = add_padding(grnd_truth)
flatfield = add_padding(flatfield)
mean_image = get_widefield_image(flatfield, ground_truth)

fft_image = vec(abs.(fftshift(fft(ifftshift(mean_image[
						padding_size+1:end-padding_size,
                             padding_size+1:end-padding_size]))))) .+eps()
logpdf(Dirichlet(modulation_transfer_function_vectorized),
                                fft_image ./ sum(fft_image))




grid_physical_1D_x = dx .* collect(-(raw_image_size_x/2 + padding_size):
                                (img_size_x/2 + padding_size - 1)) # in micrometers
grid_physical_1D_y = dx .* collect(-(img_size_y/2 + padding_size):
                                (img_size_y/2 + padding_size - 1)) # in micrometers


function incoherent_PSF(x_c::Vector{Float64}, x_e::Vector{Float64})
    return exp(-norm(x_c-x_e)^2/(2.0*sigma^2)) /
        (sqrt(2.0*pi) * sigma)^(size(x_e)[1])
end

function FFT_incoherent_PSF()
    psf_on_grid = zeros(img_size_x+2*padding_size, img_size_y+2*padding_size)
    for i in 1:img_size_x + padding_size
        for j in 1:img_size_y + padding_size
            x_e::Vector{Float64} = [grid_physical_1D_x[i], grid_physical_1D_y[j]]
            psf_on_grid[i, j] =  incoherent_PSF([0.0, 0.0], x_e)
        end
    end
    return fft(ifftshift(psf_on_grid))
end

FFT_point_spread_function =  FFT_incoherent_PSF()

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

file_name = string(working_directory,
				"ground_truth.tif")
grnd_truth = TiffImages.load(file_name)
grnd_truth = Float64.(grnd_truth)

file_name = string(working_directory,
				"poisson_image.tif")
grnd_truth_poisson = TiffImages.load(file_name)
grnd_truth_poisson = Float64.(grnd_truth_poisson)



inferred_density = zeros(img_size_x+2*padding_size, img_size_y+2*padding_size)
inferred_density[padding_size+1:end-padding_size,
		 padding_size+1:end-padding_size]=grnd_truth[:, :]

inferred_shot_noise_image = zeros(img_size_x+2*padding_size, img_size_y+2*padding_size)
inferred_shot_noise_image[padding_size+1:end-padding_size,
		 padding_size+1:end-padding_size]=grnd_truth_poisson[:, :]
mean_im = get_widefield_image(flatfield, inferred_density)

heatmap(mean_im .- mean_image)
