# B-SIM
Physically Accurate Structured Illumination Microscopy Reconstruction


B-SIM is a structured illumination micrsocopy reconstruction algorithm that accurately incorporates all sources of noise in the data and provides stricly positive solutions without arbitrary smoothness constraints. It implements Markov chain Monte Carlo (MCMC) algorithms to learn probability distributions over the main function of interest: fluorescence profile (product of fluorophore density and quantum yield given illumination). These tools can be used in a simple plug and play manner. Check the following paper to see details of all the mathematics involved in the development of B-SIM:

https://biorxiv.org/cgi/content/short/2022.07.20.500887v1

## Julia Installation

All the codes are written in Julia language for high performance/speed (similar to C and Fortran) and its open-source/free availability. Julia also allows easy parallelization of all the codes. To install julia, please download and install julia language installer from their official website (see below) for your operating system or use your package manager. The current version of the code has been successfully tested on linux (Ubuntu 22.04), macOS 12 (TEST), and Windows.

https://julialang.org/

Like python, Julia also has an interactive environment (commonly known as REPL) which can be used to add packages and perform simple tests as shown in the picture below.

![Screenshot from 2022-08-01 13-00-40](https://user-images.githubusercontent.com/87823118/182234995-db174ea5-3157-4b8c-98b9-dd0aeabc4399.png)


In Windows, this interactive environment can be started by clicking on the Julia icon on Desktop that is created upon installation or by going into the programs menu directly. We use this environment to install some essential julia packages that help simplify linear algebra and statistical calculations, and plotting. To add these packages via julia REPL, first enter the julia package manager by executing "]" command in the REPL. Then simply execute the following command to install all these packages at the same time. 

```add Random, SpecialFunctions, Distributions, Distributed, LinearAlgebra, Statistics, Plots, HDF5, TiffImages```



## Test Example

Complex programs like B-SIM require scripts for better organization instead of typing functions into the REPL for every run. B-SIM is currently organized into two scripts. First script "BSIM.jl" contains all the functions performing SIM reconstruction and the second script "input_parameters.jl" defines all the input parameters needed to perform reconstruction (see the image below). 

![Screenshot from 2023-11-03 10-53-58](https://github.com/ayushsaurabh/B-SIM/assets/87823118/8480b736-ed19-4d12-8376-46a93bbdbe4b)

These parameters define the shape of the microscope point spread function (numerical aperture, magnification, light wavelength), camera noise (gain, CCD sensitivity, readout), directory (folder) where files are located, file name, parallelization and inference settings. Using the settings in the image above, we here provide a simple plug and play example to test the functioning of B-SIM on a personal computer. For this example, we provide two tiff files "example_1_raw_images.tif" and "example_1_illumination_patterns.tif" containing 9 sinuosidal patterns and corresponding raw images. With the default settings in the image above, the code divides the image into 16 sub-images of equal size (a 4x4 grid). The sub-images are then sent to each processor and inference is performed on the fluorescence profile. The number of processors can be changed if running on a more powerful computer.

## A Brief Description of the Sampler

The samplers here execute a Markov Chain Monte Carlo (MCMC) algorithm (Gibbs) where samples for fluorescence profile at each pixel are generated sequentially from their corresponding probability distributions (posterior). First, the sampler creates/initiates arrays to store all the samples and posterior probability values. Next, new samples are then iteratively proposed using proposal (normal) distributions for each pixel, to be accepted or rejected by the Metropolis-Hastings step if direct sampling is not available. If accepted, the proposed sample is stored in the arrays otherwise the previous sample is stored at the same MCMC iteraion. 

The variance of the proposal distribution typically decides how often proposals are accepted/rejected. A larger covariance or movement away from the previous sample would lead to a larger change in likelihood/posterior values. Since the sampler prefers movement towards high probability regions, a larger movement towards low probability regions would lead to likely rejection of the sample compared to smaller movement.

The collected samples can then be used to compute statistical quantities and plot probability distributions. The plotting function used by the sampler in this code allows monitoring of posterior values and current fluorescence profile sample.

As mentioned before, sampler prefers movement towards higher probability regions of the posterior distribution. This means that if parameters are initialized in low probability regions of the posterior, which is typically the case, the posterior would appear to increase initially for many iterations (hundreds to thousands depending on the complexity of the model). This initial period is called burn-in. After burn-in, convergence is achieved where the posterior would typically fluctuate around some mean/average value. The convergence typically indicates that the sampler has reached the maximum of the posterior distribution (the most probability region), that is, sampler generates most samples from higher probability region. In other words, given a large collection of samples, the probability density in a region of parameter space is proportional to the number of samples collected from that region. 
 
All the samples collected during burn-in are usually ignored when computing statistical properties and presenting the final posterior distribution. 
