function get_input_raw_images()
 	file_name = string(working_directory,
				"raw_images.tif")
   	imgs = TiffImages.load(file_name)
#   img = reinterpret(UInt16, img)
   	imgs = Float64.(imgs)

    return imgs
end
const input_raw_images::Array{Float64} = get_input_raw_images()

const raw_img_size_x::Int64 = size(input_raw_images)[1]
const raw_img_size_y::Int64 = size(input_raw_images)[2]
const n_patterns::Int64 = size(input_raw_images)[3]


function get_input_patterns()
 	file_name = string(working_directory,
				"patterns.tif")
   	imgs = TiffImages.load(file_name)
#   img = reinterpret(UInt16, img)
   	imgs = Float64.(imgs)

    return imgs
end
const input_patterns::Array{Float64} = get_input_patterns()


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
		error_map = sqrt.(variance_map)

        file_name = string(working_directory,
                            "gain_map.tif")
        gain_map = TiffImages.load(file_name)
        gain_map = Float64.(gain_map)
    else

        offset_map = offset .* ones(raw_img_size_x, raw_img_size_y)
        gain_map = gain .* ones(raw_img_size_x, raw_img_size_y)
        error_map = noise .* ones(raw_img_size_x, raw_img_size_y)

    end

    return offset_map, error_map, gain_map
end

const offset_map::Matrix{Float64}, error_map::Matrix{Float64}, gain_map::Matrix{Float64} = get_camera_calibration_data()

function add_padding_reflective_BC_raw(input_imgs::Array{Float64},
					padding_value::Int64)


	size_x::Int64 = size(input_imgs)[1]
	size_y::Int64 = size(input_imgs)[2]

	imgs::Array{Float64} = zeros(3*size_x, 3*size_y, n_patterns)
	imgs[size_x+1:end-size_x,
			size_y+1:end-size_y, :] .= input_imgs 
	imgs[1:size_x,
			size_y+1:end-size_y, :] .= input_imgs[end:-1:1, :, :] 
	imgs[end-size_x+1:end,
			size_y+1:end-size_y, :] .= input_imgs[end:-1:1, :, :] 
	imgs[size_x+1:end-size_x,
			1:size_y, :] .= input_imgs[:, end:-1:1, :] 
	imgs[size_x+1:end - size_x,
        		end-size_y+1:end, :] .= input_imgs[:, end:-1:1, :] 
	imgs[1:size_x, 1:size_y, :] .= input_imgs[end:-1:1, end:-1:1, :] 
	imgs[1:size_x, end-size_y+1:end, :] .= input_imgs[end:-1:1, end:-1:1, :] 
	imgs[end-size_x+1:end, 1:size_y, :] .= input_imgs[end:-1:1, end:-1:1, :] 
	imgs[end-size_x+1:end, end-size_y+1:end, :] .= input_imgs[end:-1:1, end:-1:1, :] 

	return imgs[size_x-padding_value+1:2*size_x+padding_value, size_y-padding_value+1:2*size_y+padding_value, :]
end
function add_padding_reflective_BC_calibration(input_img::Matrix{Float64}, padding_value::Int64)


	size_x::Int64 = size(input_img)[1]
	size_y::Int64 = size(input_img)[2]

	img::Matrix{Float64} = zeros(3*size_x, 3*size_y)
	img[size_x+1:end-size_x,
			size_y+1:end-size_y] .= input_img 
	img[1:size_x,
			size_y+1:end-size_y] .= input_img[end:-1:1, :] 
	img[end-size_x+1:end,
			size_y+1:end-size_y] .= input_img[end:-1:1, :] 
	img[size_x+1:end-size_x,
			1:size_y] .= input_img[:, end:-1:1] 
	img[size_x+1:end - size_x,
        end-size_y+1:end] .= input_img[:, end:-1:1] 
	img[1:size_x,1:size_y] .= input_img[end:-1:1, end:-1:1] 
	img[1:size_x,end-size_y+1:end] .= input_img[end:-1:1, end:-1:1] 
	img[end-size_x+1:end,1:size_y] .= input_img[end:-1:1, end:-1:1] 
	img[end-size_x+1:end,end-size_y+1:end] .= input_img[end:-1:1, end:-1:1] 

	return img[size_x-padding_value+1:2*size_x+padding_value, size_y-padding_value+1:2*size_y+padding_value]
end


function apply_reflective_BC_object!(object::Matrix{Float64}, intermediate_object::Matrix{Float64})

	size_x::Int64 = size(object)[1] - 2*padding_size
	size_y::Int64 = size(object)[2] - 2*padding_size

	intermediate_object[size_x+1:end-size_x,
			    size_y+1:end-size_y] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size] 
	intermediate_object[1:size_x,
			    size_y+1:end-size_y] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, :] 
	intermediate_object[end-size_x+1:end,
			    size_y+1:end-size_y] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, :] 
	intermediate_object[size_x+1:end-size_x,
			    1:size_y] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size][:, end:-1:1] 
	intermediate_object[size_x+1:end - size_x,
			    end-size_y+1:end] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size][:, end:-1:1] 
	intermediate_object[1:size_x,1:size_y] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, end:-1:1] 
	intermediate_object[1:size_x,end-size_y+1:end] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, end:-1:1] 
	intermediate_object[end-size_x+1:end,1:size_y] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, end:-1:1] 
	intermediate_object[end-size_x+1:end,end-size_y+1:end] .= object[padding_size+1:end-padding_size, padding_size+1:end-padding_size][end:-1:1, end:-1:1] 

	object .= intermediate_object[size_x-padding_size+1:2*size_x+padding_size, 
				   size_y-padding_size+1:2*size_y+padding_size]

	return nothing
end
function apply_reflective_BC_shot!(shot_noise_images::Array{Float64}, intermediate_shot::Array{Float64})

	size_x::Int64 = size(shot_noise_images)[1] - 2*half_padding_size 
	size_y::Int64 = size(shot_noise_images)[2] - 2*half_padding_size

	intermediate_shot[size_x+1:end-size_x,
			  size_y+1:end-size_y, :] .= shot_noise_images[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size, :] 
	intermediate_shot[1:size_x,
			size_y+1:end-size_y, :] .= shot_noise_images[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size, :][end:-1:1, :, :] 
	intermediate_shot[end-size_x+1:end,
			size_y+1:end-size_y, :] .= shot_noise_images[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size, :][end:-1:1, :, :] 
	intermediate_shot[size_x+1:end-size_x,
			1:size_y, :] .= shot_noise_images[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size, :][:, end:-1:1, :] 
	intermediate_shot[size_x+1:end - size_x,
        		end-size_y+1:end, :] .= shot_noise_images[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size, :][:, end:-1:1, :] 
	intermediate_shot[1:size_x, 1:size_y, :] .= shot_noise_images[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size, :][end:-1:1, end:-1:1, :] 
	intermediate_shot[1:size_x, end-size_y+1:end, :] .= shot_noise_images[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size, :][end:-1:1, end:-1:1, :] 
	intermediate_shot[end-size_x+1:end, 1:size_y, :] .= shot_noise_images[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size, :][end:-1:1, end:-1:1, :] 
	intermediate_shot[end-size_x+1:end, end-size_y+1:end, :] .= shot_noise_images[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size, :][end:-1:1, end:-1:1, :] 

	shot_noise_images .= intermediate_shot[size_x-half_padding_size+1:2*size_x+half_padding_size, 
				   size_y-half_padding_size+1:2*size_y+half_padding_size, :]

	return nothing
end

function apply_reflective_BC_bg!(bg::Matrix{Float64}, intermediate_bg::Matrix{Float64})

	size_x::Int64 = size(bg)[1] - 2*half_padding_size 
	size_y::Int64 = size(bg)[2] - 2*half_padding_size

	intermediate_bg[size_x+1:end-size_x,
			  size_y+1:end-size_y] .= bg[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size] 
	intermediate_bg[1:size_x,
			size_y+1:end-size_y] .= bg[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size][end:-1:1, :] 
	intermediate_bg[end-size_x+1:end,
			size_y+1:end-size_y] .= bg[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size][end:-1:1, :] 
	intermediate_bg[size_x+1:end-size_x,
				1:size_y] .= bg[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size][:, end:-1:1] 
	intermediate_bg[size_x+1:end - size_x,
        		end-size_y+1:end] .= bg[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size][:, end:-1:1] 
	intermediate_bg[1:size_x, 1:size_y] .= bg[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size][end:-1:1, end:-1:1] 
	intermediate_bg[1:size_x, end-size_y+1:end] .= bg[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size][end:-1:1, end:-1:1] 
	intermediate_bg[end-size_x+1:end, 1:size_y] .= bg[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size][end:-1:1, end:-1:1] 
	intermediate_bg[end-size_x+1:end, end-size_y+1:end] .= bg[half_padding_size+1:end-half_padding_size,
								       half_padding_size+1:end-half_padding_size][end:-1:1, end:-1:1] 

	bg .= intermediate_bg[size_x-half_padding_size+1:2*size_x+half_padding_size, 
				   size_y-half_padding_size+1:2*size_y+half_padding_size]

	return nothing
end



#function add_padding_raw_images(raw_imgs::Array{Float64}, 
#                                scaling_factor::Float64, 
#                                padding_value::Int64)
#
#	imgs::Array{Float64} = scaling_factor .* ones(size(raw_imgs)[1]+2*padding_value, size(raw_imgs)[2]+2*padding_value, size(raw_imgs)[3])
#    	for i in 1:size(raw_imgs)[3]
#	    imgs[padding_value+1:end-padding_value,
#			padding_value+1:end-padding_value, i] .= raw_imgs[:, :, i]
#    	end
#	return imgs
#end

#const raw_images_with_padding::Array{Float64} = add_padding_raw_images(input_raw_images, 0.0, half_padding_size)
#const patterns_with_padding::Array{Float64} = add_padding_raw_images(input_patterns, 0.0, padding_size)
const raw_images_with_padding::Array{Float64} = add_padding_reflective_BC_raw(input_raw_images, half_padding_size)
const patterns_with_padding::Array{Float64} = add_padding_reflective_BC_raw(input_patterns, padding_size)


const sim_img_size_x::Int64 = 2*raw_img_size_x
const sim_img_size_y::Int64 = 2*raw_img_size_y 


#function add_padding(input_map::Matrix{Float64}, scaling_factor::Float64, padding_value::Int64)
#
#	img::Matrix{Float64} = scaling_factor .* 
#                ones(size(input_map)[1]+2*padding_value, size(input_map)[2]+2*padding_value)
#	img[padding_value+1:end-padding_value,
#			padding_value+1:end-padding_value] = input_map[:, :]
#	return img
#end
#
#intermediate_img = add_padding(gain_map, gain, internal_padding_size)
#const gain_map_with_padding::Matrix{Float64} = add_padding(intermediate_img, gain, half_padding_size)
#
#intermediate_img = add_padding(offset_map, offset, internal_padding_size)
#const offset_map_with_padding::Matrix{Float64} = add_padding(intermediate_img, offset, half_padding_size)
#
#intermediate_img = add_padding(error_map, noise, internal_padding_size)
#const error_map_with_padding::Matrix{Float64} = add_padding(intermediate_img, noise, half_padding_size)
#
const gain_map_with_padding::Matrix{Float64} = add_padding_reflective_BC_calibration(gain_map, half_padding_size)
const offset_map_with_padding::Matrix{Float64} = add_padding_reflective_BC_calibration(offset_map, half_padding_size)
const error_map_with_padding::Matrix{Float64} = add_padding_reflective_BC_calibration(error_map, half_padding_size)



const median_photon_count = 
		median(abs.((input_raw_images[:, :, 1] .- median(offset_map_with_padding)) ./ (median(gain_map_with_padding))))

const sub_raw_img_size_x::Int64 = raw_img_size_x/n_procs_per_dim_x 
const sub_raw_img_size_y::Int64 = raw_img_size_y/n_procs_per_dim_y
const sub_sim_img_size_x::Int64 = sim_img_size_x/n_procs_per_dim_x 
const sub_sim_img_size_y::Int64 = sim_img_size_y/n_procs_per_dim_y

intermediate_img = 0
GC.gc()
