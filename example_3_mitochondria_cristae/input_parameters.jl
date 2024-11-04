const working_directory = string(pwd(), "/") 

# Optical Parameters
const psf_type = "airy_disk" 
const numerical_aperture::Float64 = 1.3
const magnification::Float64 = 100.0
const light_wavelength::Float64 = 0.569# In micrometers
const abbe_diffraction_limit::Float64 = light_wavelength/(2*numerical_aperture) # optical resolution in micrometers
const f_diffraction_limit::Float64 = 1/abbe_diffraction_limit # Diffraction limit in k-space
const sigma::Float64 = sqrt(2.0)/(2.0*pi)*light_wavelength/numerical_aperture # Standard Deviation in PSF

# Camera Parameters
const camera_pixel_size::Float64 = 6.5 # In micrometers
const physical_pixel_size::Float64 = camera_pixel_size/magnification #in micrometers
const dx::Float64 = 0.5 * physical_pixel_size #physical grid spacing
const gain::Float64 = 1.957
const offset::Float64 = 100.0
const noise::Float64 = 2.3
const noise_maps_available = false

# Inference Parameters
const padding_size::Int64 = 12*ceil(abbe_diffraction_limit/physical_pixel_size) 
const half_padding_size::Int64 = padding_size/2
const quarter_padding_size::Int64 = padding_size/4
const internal_padding_size::Int64 = 0

const covariance_object::Float64 = 0.5
const conc_parameter::Float64 = 1.0

const total_draws::Int64 = 100000
const chain_burn_in_period::Int64 = 4000
const chain_starting_temperature::Int64 = 1000.0
const chain_time_constant::Float64 = 200.0

const annealing_starting_temperature::Float64 =100.0
const annealing_time_constant::Float64 = 30.0
const annealing_burn_in_period::Int64 = 300
const annealing_frequency::Int64 = annealing_burn_in_period + 50
const averaging_frequency::Int64 = 10

# Plotting Options
const plotting = true
const plotting_frequency::Int64 = 10

# Number of Processors Available to use 
n_procs_per_dim_x::Int64 = 4
n_procs_per_dim_y::Int64 = 4
