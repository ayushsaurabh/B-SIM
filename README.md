# B-SIM
A Structured Illumination Microscopy Reconstruction Code with Spatial-Domain Noise Propagation


B-SIM is a structured illumination micrsocopy reconstruction algorithm that accurately incorporates all sources of noise in the data and provides stricly positive solutions without arbitrary smoothness constraints or ad-hoc parameter tuning. It implements Markov chain Monte Carlo (MCMC) algorithms to learn probability distributions over the main function of interest: fluorescence intensity map (product of fluorophore density and quantum yield given illumination). These tools can be used in a simple plug and play manner. Check the following paper to see details of all the mathematics involved in the development of B-SIM:

https://biorxiv.org/cgi/content/short/2022.07.20.500887v1

## Julia Installation

**System Requirements: B-SIM is a parallelized code and has been fully tested on modern desktops and laptops with at least 4 processor cores.  In addition to about twice the memory required to load SIM raw images, illumination patterns, and SIM image globally, B-SIM typically requests 300-500 MB of RAM to load Julia packages on each processor core.**

All the codes are written in Julia language for high performance/speed (similar to C and Fortran) and its open-source/free availability. Julia also allows easy parallelization of all the codes. To install julia, please download and install julia language installer from their official website (see below) for your operating system or use your package manager. The current version of the code has been successfully tested on linux (Ubuntu 22.04), macOS 12, and Windows.

https://julialang.org/

Like python, Julia also has an interactive environment (commonly known as REPL) which can be used to add packages and perform simple tests as shown in the picture below.


![Screenshot from 2023-11-08 14-36-41](https://github.com/ayushsaurabh/B-SIM/assets/87823118/05bdffb9-6857-4209-9d8d-97cedd3a3578)


In Windows, this interactive environment can be started by clicking on the Julia icon on Desktop that is created upon installation or by going into the programs menu directly. On Linux or macOS machines, julia REPL can be accessed by simply typing julia in the terminal. We use this environment to install some essential julia packages that help simplify linear algebra and statistical calculations, and plotting. To add these packages via julia REPL, **first enter the julia package manager by executing "]" command in the REPL**. Then simply execute the following command to install all these packages at the same time. 

```add Distributed, Random, SpecialFunctions, Distributions, LinearAlgebra, Statistics, Plots, HDF5, TiffImages, FixedPointNumbers, FFTW```


![Screenshot from 2023-11-08 14-40-31](https://github.com/ayushsaurabh/B-SIM/assets/87823118/27ffde07-7eb8-40a5-871b-cc4ea0e34859)

**To get out of the package manager, simply hit the backspace key.**

## Test Example

Complex programs like B-SIM require scripts for better organization instead of typing functions into the REPL for every run. B-SIM is currently organized into two scripts. First script "B-SIM.jl" contains all the functions performing SIM reconstruction and the second script "input_parameters.jl" defines all the input parameters needed to perform reconstruction (see the image below).


![Screenshot from 2023-12-08 16-09-53](https://github.com/ayushsaurabh/B-SIM/assets/87823118/31d76292-619f-4f97-a11e-b25ed25f8158)



These parameters define the shape of the microscope point spread function (numerical aperture, magnification, light wavelength), camera noise (gain, CCD sensitivity, readout), directory (folder) where files are located, file name, parallelization and inference settings. 

Using the settings in the image above, we here provide a simple plug and play example to test the functioning of B-SIM on a personal computer. For this example, we provide three tiff files "raw_images_line_pairs_84x84_500nm_highSNR.tif", "illumination_patterns_line_pairs_168x168_500nm_highSNR.tif", and "ground_truth_line_pairs_168x168_500nm_highSNR.tif" containing 9 sinuosidal patterns and corresponding raw images as well as the ground truth. 

Currently, B-SIM only accepts square images but can be easily modified to accept rectangular images. Furthermore, the current version assumes a Gaussian point spread function but can be modified easily to incorporate any other shape.

With the settings in the image above, the code divides the image into 16 sub-images of equal size (a 4x4 grid). The sub-images are then sent to each processor and inference is performed on the fluorescence intensity map. The number of processors can be changed if running on a more powerful computer by changing "n_procs_per_dim" parameter, which is set to 2 by default.

To run this example, we suggest putting B-SIM scripts and the input tif files in the same folder/directory. Next, if running on a Windows machine, first confirm the current folder that julia is being run from by executing the following command in the REPL:

```pwd()```

**Please note here that Windows machines use backslashes "\\" to describe folder paths unlike Linux and macOS where forward slashes "/" are used.** Appropriate modifications therefore must be made to the folder paths. Now, if the output of the command above is different from the path containing the scripts and tiff files, the current path can be changed by executing the following command:

```cd("/home/username/B-SIM/")```

B-SIM code can now be executed by simply importing the "B-SIM.jl" in the REPL as shown in the picture below

```include("B-SIM.jl")```


![Screenshot from 2023-11-08 14-10-07](https://github.com/ayushsaurabh/B-SIM/assets/87823118/cf5bf788-ab49-48f6-b2e2-586382eb1c0f)


On a linux or macOS machine, the "B-SIM.jl" script can be run directly from the terminal after entering the B-SIM directory and executing the following command:

```julia B-SIM.jl```

**WARNING: Please note that when running the code through the REPL, restart the REPL if B-SIM throws an error. Every execution of B-SIM adds processors to the julia REPL and processor ID or label increases in value. To make sure that processor labels always start at 1, we suggest avoiding restarting B-SIM in the same REPL.**

Now, B-SIM is a fully parallelized code and starts execution by first adding the required number of processors. Next, all the input tif files are imported and divided according to the parallelization grid (4x4 by default). The sub-images are then sent to each processor. All the functions involved in SIM reconstruction are compiled next. Finally, the sampler starts and with each iteration outputs the log(posterior) values and a temperature parameter that users are not required to modify (see picture below).


![Screenshot from 2023-11-08 14-20-59](https://github.com/ayushsaurabh/B-SIM/assets/87823118/28210b7f-3fdb-40b7-9929-5cc7f2ca9925)


Depending on whether plotting option is chosen to be ```true``` or ```false``` in the "input_parameters.jl" file, the code also generates a plot showing the the log(posterior), one of the input raw images, the intermediate shot noise image, ground truth, current sample, and a mean of the previous samples (depending on averaging frequency) as shown in the picture below.


![Screenshot from 2023-11-08 14-27-24](https://github.com/ayushsaurabh/B-SIM/assets/87823118/8ecfc77e-ac1f-4be9-b8ef-5d4b7da5ce0f)


Finally, as samples are collected, B-SIM saves intermediate samples and analysis data onto the hard drive in the HDF5 format with file names that look like "mcmc_output_...1000.h5". The save frequency can be modified by changing a few inference parameters in the "input_parameters.jl" file: "initial_burn_in_period" which is set based on when the sampler converges for the **first time**; simulated annealing is restarted at regular intervals set by the parameter "annealing_frequency"; simulated annealing starts with temperature set by "annealing_starting_temperature" and then the temperature decays exponentially with time constant set by "annealing_time_constant"; samples to be averaged are collected after the "annealing_burn_in_period" during which the sampler converges after increasing the temperature; and lastly, samples are collected at the "averaging_frequency" after the annealing burn-in period. Use of simulated annealing here helps uncorrelate the chain of samples by smoothing and widening the posterior at intermediate iterations by raising temperature, allowing the sampler to easily move far away from the current sample. Based on these parameters, the samples are saved whenever the following conditions are satisfied: 

```
if (draw >= initial_burn_in_period) &&
   (draw % annealing_frequency >= annealing_burn_in_period) &&
      (draw % averaging_frequency == 0)

```
where % indicates remainder upon division. For instance, using the default settings in the "input_parameters.jl" file shown above, the samples will be saved at iterations:

``` 
4420, 4440, 4460, ..., 4980,
5420, 5440, 5460, ..., 5980,
6420, 6440, 6460, ..., 6980, ...
```


## A Brief Description of the Sampler

The sampler here execute a Markov Chain Monte Carlo (MCMC) algorithm (Gibbs) where samples for fluorescence intensity at each pixel are generated sequentially from their corresponding conditional probability distributions (posterior). First, the sampler creates/initiates arrays to store all the samples and joint posterior probability values. Next, new samples are then iteratively proposed using proposal (normal) distributions for each pixel, to be accepted or rejected by the Metropolis-Hastings step if direct sampling is not available. If accepted, the proposed sample is stored in the arrays otherwise the previous sample is stored at the same MCMC iteraion. 

The variance of the proposal distribution typically decides how often proposals are accepted/rejected. A larger covariance or movement away from the previous sample would lead to a larger change in likelihood/posterior values. Since the sampler prefers movement towards high probability regions, a larger movement towards low probability regions would lead to likely rejection of the sample compared to smaller movement.

The collected samples can then be used to compute statistical quantities and plot probability distributions. The plotting function used by the sampler in this code allows monitoring of posterior values and current fluorescence intensity sample.

As mentioned before, sampler prefers movement towards higher probability regions of the posterior distribution. This means that if parameters are initialized in low probability regions of the posterior, which is typically the case, the posterior would appear to increase initially for many iterations (hundreds to thousands depending on the complexity of the model). This initial period is called burn-in. After burn-in, convergence is achieved where the posterior would typically fluctuate around some mean/average value. The convergence typically indicates that the sampler has reached the maximum of the posterior distribution (the most probable region), that is, sampler generates most samples from higher probability region. In other words, given a large collection of samples, the probability density in a region of parameter space is proportional to the number of samples collected from that region. 
 
All the samples collected during burn-in are usually ignored when computing statistical properties and presenting the final posterior distribution. 
