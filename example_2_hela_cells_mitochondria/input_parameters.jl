const working_directory = string(pwd(), "/") 

# Optical Parameters
const psf_type = "gaussian" 
const numerical_aperture::Float64 = 1.3
const magnification::Float64 = 100.0
const light_wavelength::Float64 = 0.660# In micrometers
const abbe_diffraction_limit::Float64 = light_wavelength/(2*numerical_aperture) # optical resolution in micrometers
const f_diffraction_limit::Float64 = 1/abbe_diffraction_limit # Diffraction limit in k-space
const sigma::Float64 = sqrt(2.0)/(2.0*pi)*light_wavelength/numerical_aperture # Standard Deviation in PSF

# Camera Parameters
const camera_pixel_size::Float64 = 6.5 # In micrometers
const physical_pixel_size::Float64 = camera_pixel_size/magnification #in micrometers
const dx::Float64 = 0.5 * physical_pixel_size #physical grid spacing
const gain::Float64 = 2.0
const offset::Float64 = 100.0
const noise::Float64 = 2.0
const noise_maps_available = true

# Inference Parameters: Spatial Domain Padding to minimize wrap-around artifacts
const padding_size::Int64 = 8*ceil(abbe_diffraction_limit/physical_pixel_size) 
const half_padding_size::Int64 = padding_size/2
const quarter_padding_size::Int64 = padding_size/4
const internal_padding_size::Int64 = 0

# Step size for proposing new Monte Carlo samples. Value of 0.5 seems optimal
const covariance_object::Float64 = 0.5

# Dirichlet Prior Concentration Parameter. Higher value increases the probability of samples
# that resemble MTF more closely, thereby penalizing non-similar samples more heavily
const conc_parameter::Float64 = 1.0

# Number of Processors to add = n_procs_per_dim_x * n_procs_per_dim_y
n_procs_per_dim_x::Int64 = 2
n_procs_per_dim_y::Int64 = 2


const total_draws::Int64 = 100000
const initial_burn_in_period::Int64 = 0	
const annealing_starting_temperature::Float64 = 100000.0
const annealing_time_constant::Float64 = 70.0
const annealing_burn_in_period::Int64 = 1000
const annealing_frequency::Int64 = annealing_burn_in_period + 250
const averaging_frequency::Int64 = 50
const plotting_frequency::Int64 = 10
