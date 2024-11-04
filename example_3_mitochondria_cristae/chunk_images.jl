const raw_img_size_x::Int64 = @fetchfrom 1 raw_img_size_x
const raw_img_size_y::Int64 = @fetchfrom 1 raw_img_size_y
const sub_raw_img_size_x::Int64 = @fetchfrom 1 sub_raw_img_size_x
const sub_raw_img_size_y::Int64 = @fetchfrom 1 sub_raw_img_size_y

const sim_img_size_x::Int64 = @fetchfrom 1 sim_img_size_x
const sim_img_size_y::Int64 = @fetchfrom 1 sim_img_size_y
const sub_sim_img_size_x::Int64 = @fetchfrom 1 sub_sim_img_size_x
const sub_sim_img_size_y::Int64 = @fetchfrom 1 sub_sim_img_size_y

const n_patterns::Int64 = @fetchfrom 1 n_patterns



# Cartesian coordinates for each processor
const i_procs::Int64 = (myid()-2) % n_procs_per_dim_x
const j_procs::Int64 = (myid()-2 - i_procs)/n_procs_per_dim_x

# Boundary coordinates of chunks or sub raw images in x an y direction
const im_raw::Int64 = i_procs*sub_raw_img_size_x + 1 #+ internal_padding_size
const ip_raw::Int64 = 2*half_padding_size + (i_procs+1)*sub_raw_img_size_x #+ internal_padding_size
const jm_raw::Int64 = j_procs*sub_raw_img_size_y + 1 #+ internal_padding_size
const jp_raw::Int64 = 2*half_padding_size + (j_procs+1)*sub_raw_img_size_y #+ internal_padding_size

const im_sim::Int64 = i_procs*sub_sim_img_size_x + 1 #+ internal_padding_size
const ip_sim::Int64 = 2*padding_size + (i_procs+1)*sub_sim_img_size_x #+ internal_padding_size
const jm_sim::Int64 = j_procs*sub_sim_img_size_y + 1 #+ internal_padding_size
const jp_sim::Int64 = 2*padding_size + (j_procs+1)*sub_sim_img_size_y #+ internal_padding_size


#if i_procs == 0
#	im_raw = i_procs*sub_raw_img_size_x + 1
#end
#if i_procs == n_procs_per_dim_x - 1
#	ip_raw = ip_raw + internal_padding_size 
#end
#if j_procs == 0
#	jm_raw = j_procs*sub_raw_img_size_y + 1
#end
#if j_procs == n_procs_per_dim_y - 1
#	jp_raw = jp_raw + internal_padding_size 
#end

#const sub_img_size_x::Int64 = ip_raw - im_raw + 1 - 2*padding_size
#const sub_img_size_y::Int64 = jp_raw - jm_raw + 1 - 2*padding_size



function get_sub_images(imgs::Array{Float64},
				im::Int64, ip::Int64, jm::Int64, jp::Int64)
	sub_imgs::Array{Float64} = zeros(ip-im+1, jp-jm+1, n_patterns)
	for i in 1:n_patterns
		sub_imgs[:, :, i] .= imgs[im:ip, jm:jp, i]
	end
	return sub_imgs
end

function get_sub_image(img::Matrix{Float64},
				im::Int64, ip::Int64, jm::Int64, jp::Int64)
	sub_img = img[im:ip, jm:jp]
	return sub_img
end

images = @fetchfrom 1 raw_images_with_padding
const sub_raw_images::Array{Float64} = get_sub_images(images, im_raw, ip_raw, jm_raw, jp_raw)

images = @fetchfrom 1 patterns_with_padding
const sub_patterns::Array{Float64} = get_sub_images(images, im_sim, ip_sim, jm_sim, jp_sim)


image = @fetchfrom 1 gain_map_with_padding
const sub_gain_map::Matrix{Float64} = get_sub_image(image, im_raw, ip_raw, jm_raw, jp_raw)

image = @fetchfrom 1 offset_map_with_padding
const sub_offset_map::Matrix{Float64} = get_sub_image(image, im_raw, ip_raw, jm_raw, jp_raw)

image = @fetchfrom 1 error_map_with_padding
const sub_error_map::Matrix{Float64} = get_sub_image(image, im_raw, ip_raw, jm_raw, jp_raw)

image = 0
images = 0
GC.gc()
