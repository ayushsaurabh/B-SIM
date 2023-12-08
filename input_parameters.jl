const working_directory = string(pwd(),"/")

# Image Properties
const raw_image_size::Integer = 84
const raw_file_name_suffix = string("line_pairs_", raw_image_size,"x", raw_image_size,"_500nm_highSNR")
const illumination_file_name_suffix = string("line_pairs_", 2*raw_image_size,"x", 2*raw_image_size,"_500nm_highSNR")



# Optical Parameters
const numerical_aperture::Float64 = 1.49
const magnification::Float64 = 100.0
const light_wavelength::Float64 = 0.500# In micrometers
const abbe_diffraction_limit::Float64 = light_wavelength/(2*numerical_aperture) # optical resolution in micrometers
const f_diffraction_limit::Float64 = 1/abbe_diffraction_limit # Diffraction limit in k-space
const sigma::Float64 = sqrt(2.0)/(2.0*pi)*light_wavelength/numerical_aperture # Standard Deviation in PSF

# Camera Parameters
const camera_pixel_size::Float64 = 6.0 # In micrometers
const physical_pixel_size::Float64 = camera_pixel_size/magnification #in micrometers
const dx::Float64 = 0.5*physical_pixel_size #physical grid spacing
const gain::Float64 = 2.0
const offset::Float64 = 100.0
const noise::Float64 = 4.0
const noise_maps_available = false

# Inference Parameters
const ghost_size::Integer = 24 # Always choose numbers divisible by 4
const half_ghost_size::Integer = ghost_size/2
const covariance_gt::Float64 = 2.0
const gamma_prior_scale::Float64 = 10.0
const gamma_prior_shape::Float64 = 0.1

const total_draws::Integer = 10000
const initial_burn_in_period::Integer = 4000
const annealing_frequency::Integer = 1000
const annealing_time_constant::Float64 = 20.0
const annealing_starting_temperature::Float64 = 10.0
const annealing_burn_in_period::Integer = 400
const averaging_frequency::Integer = 20


# Parallelization Parameters
const n_procs_per_dim::Integer = 2

# Plotting Options
const plotting = true
const plotting_frequency = 50
const posterior_moving_window_size = 400



