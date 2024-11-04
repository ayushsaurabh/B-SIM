function save_data(current_draw::Integer,
    mcmc_log_posterior::Vector{Float64},
    object::Matrix{Float64},
    shot_noise_images::Array{Float64},
    bg::Matrix{Float64},
    object_mean::Matrix{Float64},
    averaging_counter::Float64)

	grays = convert.(Gray{Float64}, 
		bg[half_padding_size+1:half_padding_size+raw_img_size_x, 
			half_padding_size+1:half_padding_size+raw_img_size_y])
	img = TiffImages.DenseTaggedImage(grays)
	TiffImages.save(string(working_directory,
			"inferred_bg_", averaging_counter,".tif"), img)
	
	grays = convert.(Gray{Float64}, 
		object[padding_size+1:padding_size+sim_img_size_x, 
			padding_size+1:padding_size+sim_img_size_y])
	img = TiffImages.DenseTaggedImage(grays)
	TiffImages.save(string(working_directory,
			"inferred_object_", averaging_counter,".tif"), img)
	
	grays = convert.(Gray{Float64}, 
		object_mean[padding_size+1:padding_size+sim_img_size_x, 
			padding_size+1:padding_size+sim_img_size_y])
	img = TiffImages.DenseTaggedImage(grays)
	TiffImages.save(string(working_directory,
			"mean_inferred_object_", averaging_counter,".tif"), img)
	
	
	return nothing
end

if plotting == true
	function plot_data(current_draw::Int64, 
			object::Matrix{Float64}, 
			mean_object::Matrix{Float64}, 
			shot_noise_images::Array{Float64}, 
			bg::Matrix{Float64}, 
			log_posterior::Vector{Float64})
	
		plot_object  = heatmap(view(object, 
				padding_size+1:padding_size+sim_img_size_x, 
				padding_size+1:padding_size+sim_img_size_y), 
				legend=true, c=:grays, yflip=true, title = "Current Sample")
		plot_mean_object  = heatmap(view(mean_object, 
				padding_size+1:padding_size+sim_img_size_x, 
				padding_size+1:padding_size+sim_img_size_y), 
				legend=true, c=:grays, yflip=true, title = "Mean")
		plot_shot  = heatmap(view(shot_noise_images,
				half_padding_size+1:half_padding_size+raw_img_size_x, 
				half_padding_size+1:half_padding_size+raw_img_size_y, 1), 
				legend=true, c=:grays, yflip=true, title = "Shot Noise Image")
		plot_bg  = heatmap(view(bg,
				half_padding_size+1:half_padding_size+raw_img_size_x, 
				half_padding_size+1:half_padding_size+raw_img_size_y, 1), 
				legend=true, c=:grays, yflip=true, title = "Unmodulated Background")
		
		plot_post  = plot(view(log_posterior,1:current_draw), legend=false, c=:grays, title = "log(Posterior)")
		plot_raw  = heatmap(view(raw_images_with_padding, 
				half_padding_size+1:half_padding_size+raw_img_size_x, 
				half_padding_size+1:half_padding_size+raw_img_size_y, 1), 
				legend=true, c=:grays, yflip=true, title = "A Raw Image")
		
		display(plot(plot_raw, plot_bg, plot_post, plot_object, plot_mean_object,  layout = (2, 3), size=(3000, 1300) ))
	
	
		return nothing
	end
end
