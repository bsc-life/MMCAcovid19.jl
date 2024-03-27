using Printf
using ArgParse
using Logging

using Dates
using CSV
using NPZ
using JSON
using HDF5
using DataStructures
using DelimitedFiles
using DataFrames

base_folder = ""

include(joinpath(base_folder, "src/markov_aux.jl"))
include(joinpath(base_folder, "src/markov_io.jl"))
include(joinpath(base_folder, "src/markov.jl"))

####################################################
##############   FUNCTIONS     #####################
####################################################

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--config", "-c"
            help = "config file (json file)"
            required = true
        "--data-folder", "-d"
            help = "data folder"
            required = true
        "--instance-folder", "-i"
            help = "instance folder (experiment folder)"
            required = true 
        "--export-compartments-full"
            help = "export compartments of simulations"
            action = :store_true
        "--export-compartments-time-t"
            help = "export compartments of simulations at a given time"
            default = nothing
            arg_type = Int
        "--initial-conditions"
            help = "compartments to initialize simulation. If missing, use the seeds to initialize the simulations"
            default = nothing
        "--start-date"
            help = "starting date of simulation. Overwrites the one provided in config.json"
            default = nothing
        "--end-date"
            help = "end date of simulation. Overwrites the one provided in config.json"
            default = nothing
    end

    return parse_args(s)
end



function set_compartments!(epi_params, population, 
    initial_compartments; normalize=true)

# Index of the initial condition
    t₀ = 1
    if normalize

        epi_params.ρˢᵍᵥ[:,:,t₀]  .= initial_compartments[:, :, 1] ./ population.nᵢᵍ
        epi_params.ρᴱᵍᵥ[:,:,t₀]  .= initial_compartments[:, :, 2] ./ population.nᵢᵍ
        epi_params.ρᴬᵍᵥ[:,:,t₀]  .= initial_compartments[:, :, 3] ./ population.nᵢᵍ
        epi_params.ρᴵᵍᵥ[:,:,t₀]  .= initial_compartments[:, :, 4] ./ population.nᵢᵍ
        epi_params.ρᴾᴴᵍᵥ[:,:,t₀] .= initial_compartments[:, :, 5] ./ population.nᵢᵍ
        epi_params.ρᴾᴰᵍᵥ[:,:,t₀] .= initial_compartments[:, :, 6] ./ population.nᵢᵍ
        epi_params.ρᴴᴿᵍᵥ[:,:,t₀] .= initial_compartments[:, :, 7] ./ population.nᵢᵍ
        epi_params.ρᴴᴰᵍᵥ[:,:,t₀] .= initial_compartments[:, :, 8] ./ population.nᵢᵍ
        epi_params.ρᴿᵍᵥ[:,:,t₀]  .= initial_compartments[:, :, 9] ./ population.nᵢᵍ
        epi_params.ρᴰᵍᵥ[:,:,t₀]  .= initial_compartments[:, :, 10] ./ population.nᵢᵍ

    else
    
        epi_params.ρˢᵍᵥ[:,:,t₀]  .= initial_compartments[:, :, 1] 
        epi_params.ρᴱᵍᵥ[:,:,t₀]  .= initial_compartments[:, :, 2] 
        epi_params.ρᴬᵍᵥ[:,:,t₀]  .= initial_compartments[:, :, 3] 
        epi_params.ρᴵᵍᵥ[:,:,t₀]  .= initial_compartments[:, :, 4] 
        epi_params.ρᴾᴴᵍᵥ[:,:,t₀] .= initial_compartments[:, :, 5] 
        epi_params.ρᴾᴰᵍᵥ[:,:,t₀] .= initial_compartments[:, :, 6] 
        epi_params.ρᴴᴿᵍᵥ[:,:,t₀] .= initial_compartments[:, :, 7] 
        epi_params.ρᴴᴰᵍᵥ[:,:,t₀] .= initial_compartments[:, :, 8] 
        epi_params.ρᴿᵍᵥ[:,:,t₀]  .= initial_compartments[:, :, 9] 
        epi_params.ρᴰᵍᵥ[:,:,t₀]  .= initial_compartments[:, :, 10]

    end

    epi_params.ρˢᵍᵥ[isnan.(epi_params.ρˢᵍᵥ)]   .= 0
    epi_params.ρᴱᵍᵥ[isnan.(epi_params.ρᴱᵍᵥ)]   .= 0
    epi_params.ρᴬᵍᵥ[isnan.(epi_params.ρᴬᵍᵥ)]   .= 0
    epi_params.ρᴵᵍᵥ[isnan.(epi_params.ρᴵᵍᵥ)]   .= 0
    epi_params.ρᴾᴴᵍᵥ[isnan.(epi_params.ρᴾᴴᵍᵥ)] .= 0
    epi_params.ρᴾᴰᵍᵥ[isnan.(epi_params.ρᴾᴰᵍᵥ)] .= 0
    epi_params.ρᴴᴿᵍᵥ[isnan.(epi_params.ρᴴᴿᵍᵥ)] .= 0
    epi_params.ρᴴᴰᵍᵥ[isnan.(epi_params.ρᴴᴰᵍᵥ)] .= 0
    epi_params.ρᴿᵍᵥ[isnan.(epi_params.ρᴿᵍᵥ)]   .= 0
    epi_params.ρᴰᵍᵥ[isnan.(epi_params.ρᴰᵍᵥ)]   .= 0
end

###########################################
############# FILE READING ################
###########################################

args = parse_commandline()

config_fname         = args["config"]
data_path            = args["data-folder"]
instance_path        = args["instance-folder"]
init_conditions_path = args["initial-conditions"]

@assert isfile(config_fname);
@assert isdir(data_path);
@assert isdir(instance_path);

config = JSON.parsefile(config_fname);
update_config!(config, args)

simulation_dict = config["simulation"]
data_dict       = config["data"]
epi_params_dict = config["epidemic_params"]
pop_params_dict = config["population_params"]
npi_params_dict = config["NPI"]

#########################
# Simulation output 
#########################
output_path = joinpath(instance_path, "output")
if !isdir(output_path)
    println("Creating output folder: $output_path")
    mkpath(output_path)
end

output_format    = simulation_dict["output_format"]
save_full_output = get(simulation_dict, "save_full_output", false)
save_time_step   = get(simulation_dict, "save_time_step", nothing)
init_format      = get(simulation_dict, "init_format", "netcdf")

#####################
# Initial Condition
#####################

if isnothing(init_conditions_path)
    init_conditions_path = joinpath(data_path, get(data_dict, "initial_condition_filename", nothing))
end

# use initial compartments matrix to initialize simulations
if init_format == "netcdf"
    @info "Reading initial conditions from: $(init_conditions_path)"
    initial_compartments = ncread(init_conditions_path, "data")
elseif init_format == "hdf5"
    initial_compartments = h5open(init_conditions_path, "r") do file
        read(file, "data")
    end
else
    @error "init_format must be one of : netcdf/hdf5"
end

########################################
####### VARIABLES INITIALIZATION #######
########################################

# Reading simulation start and end dates
first_day = Date(simulation_dict["first_day_simulation"])
last_day  = Date(simulation_dict["last_day_simulation"])
# Converting dates to time steps
T = (last_day - first_day).value + 1
# Array with time coordinates (dates)
T_coords  = string.(collect(first_day:last_day))

# Loading metapopulation patches info (surface, label, population by age)
metapop_data_filename = joinpath(data_path, data_dict["metapopulation_data_filename"])
metapop_df = CSV.read(metapop_data_filename, DataFrame)

# Loading mobility network
mobility_matrix_filename = joinpath(data_path, data_dict["mobility_matrix_filename"])
network_df  = CSV.read(mobility_matrix_filename, DataFrame)

# Metapopulations patches coordinates (labels)
M_coords = map(String,metapop_df[:, "id"])
M = length(M_coords)

# Coordinates for each age strata (labels)
G_coords = map(String, pop_params_dict["age_labels"])
G = length(G_coords)

####################################################
#####   INITIALIZATION OF DATA Structures   ########
####################################################

## POPULATION PARAMETERS
population       = init_pop_param_struct(G, M, G_coords, pop_params_dict, metapop_df, network_df)
## EPIDEMIC PARAMETERS 
epi_params       = init_epi_parameters_struct(G, M, T, G_coords, epi_params_dict)

@assert size(initial_compartments) == (G, M, epi_params.NumComps)

##################################################

@info "- Initializing MMCA epidemic simulations"
@info "\t- first_day_simulation = "  first_day
@info "\t- last_day_simulation = " last_day
@info "\t- G (agent class) = " G
@info "\t- M (n. of metapopulations) = "  M
@info "\t- T (simulation steps) = " T
@info "\t- N. of epi compartments = " epi_params.NumComps

@info "\t- Save full output = " save_full_output
if save_time_step !== nothing
    @info "\t- Save time step at t=" save_time_step
end

#########################################################
# Containment measures
#########################################################

# Daily Mobility reduction
kappa0_filename = get(data_dict, "kappa0_filename", nothing)

if !isnothing(kappa0_filename)
    kappa0_filename = joinpath(data_path, kappa0_filename)
    @info "- Loading κ₀ time series from $(kappa0_filename)"
    κ₀_df = CSV.read(kappa0_filename, DataFrame);
    # syncronize containment measures with simulation
    @info "- Synchronizing to dates"
    κ₀_df.time = map(x -> (x .- first_day).value + 1, κ₀_df.date)
    # Timesteps when the containment measures will be applied
    tᶜs = κ₀_df.time[:]
    # Array of level of confinement
    κ₀s = κ₀_df.reduction[:]
    # Array of premeabilities of confined households
    ϕs = Float64.(npi_params_dict["ϕs"])
    ϕs = fill(ϕs[1], length(tᶜs))
    # if length(tᶜs) != length(ϕs)
    #     if length(ϕs) == 1
    #         ϕs = fill(ϕs[1], length(tᶜs))
    #     else
    #         #error
    #         #completar vector con 0 o 1 y lanzar un warning
    #     end
    # end

        

    #ϕs = ones(Float64, length(tᶜs))
    # Array of social distancing measures
    #δs = zeros(Float64, length(tᶜs))
    δs = Float64.(npi_params_dict["δs"])
    δs = fill(δs[1], length(tᶜs))
else
    # Timesteps when the containment measures will be applied
    tᶜs = Int64.(npi_params_dict["tᶜs"])
    # Array of level of confinement
    κ₀s = Float64.(npi_params_dict["κ₀s"])
    # Array of premeabilities of confined households
    ϕs = Float64.(npi_params_dict["ϕs"])
    # Array of social distancing measures
    #δs = Float64.(npi_params_dict["δs"])
end


set_compartments!(epi_params, population, initial_compartments)

########################################################
################ RUN THE SIMULATION ####################
########################################################

run_epidemic_spreading_mmca!(epi_params, population, tᶜs, κ₀s, ϕs, δs; verbose = true )

##############################################################
################## STORING THE RESULTS #######################
##############################################################

if save_full_output
    @info "Storing full simulation output in $(output_format)"
    if output_format == "netcdf"
        filename = joinpath(output_path, "compartments_full.nc")
        @info "\t- Output filename: $(filename)"
        save_simulation_netCDF(epi_params, population, filename;G_coords=G_coords, M_coords=M_coords, T_coords=T_coords)
    elseif output_format == "hdf5"
        filename = joinpath(output_path, "compartments_full.h5")
        @info "\t- Output filename: $(filename)"
        save_simulation_hdf5(epi_params, population, filename)
    end
end

if save_time_step !== nothing
    export_compartments_date = first_day + Day(export_compartments_time_t - 1)
    filename = joinpath(output_path, "compartments_t_$(export_compartments_date).h5")
    @info "Storing compartments at single date $(export_compartments_date):"
    @info "\t- Simulation step: $(export_compartments_time_t)"
    @info "\t- filename: $(filename)"
    save_simulation_hdf5(epi_params, population, filename; 
                         export_time_t = export_compartments_time_t)
end