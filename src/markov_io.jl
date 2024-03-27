using Logging
using NetCDF

base_folder
include("markov_aux.jl")

function create_default_epi_params()
    epiparams_dict = Dict()
    epiparams_dict["scale_β"] = 0.51
    epiparams_dict["βᴬ"] = 0.046053
    epiparams_dict["βᴵ"] = 0.0903
    epiparams_dict["ηᵍ"] = [0.2747252747252747, 0.2747252747252747, 0.2747252747252747]
    epiparams_dict["αᵍ"] = [0.26595744680851063, 0.641025641025641, 0.641025641025641]
    epiparams_dict["μᵍ"] = [1.0, 0.3125, 0.3125]
    epiparams_dict["θᵍ"] = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    epiparams_dict["γᵍ"] = [0.003, 0.01, 0.08]
    epiparams_dict["ζᵍ"] = [0.12820512820512822, 0.12820512820512822, 0.12820512820512822]
    epiparams_dict["λᵍ"] = [1.0, 1.0, 1.0]
    epiparams_dict["ωᵍ"] = [0.0, 0.04, 0.3]
    epiparams_dict["ψᵍ"] = [0.14285714285714285, 0.14285714285714285, 0.14285714285714285]
    epiparams_dict["χᵍ"] = [0.047619047619047616, 0.047619047619047616, 0.047619047619047616]
    epiparams_dict["risk_reduction_dd"] = 0.0
    epiparams_dict["risk_reduction_h"] = 0.1
    epiparams_dict["risk_reduction_d"] = 0.05
    
    return epiparams_dict
end

function create_default_population_params()
    population = Dict()
    populaiont["age_labels"] = ["Y", "M", "O"]
    populaiont["C"] = [ 0.598  0.38486 0.01714 ;
                        0.244  0.721   0.0353;
                        0.1919 0.5705  0.2376
                      ]
    population["kᵍ"] = [11.8, 13.3, 6.76]
    population["kᵍ_h"] = [3.15, 3.17, 3.28]
    population["kᵍ_w"] = [1.72, 5.18, 0.0]
    population["pᵍ"] = [0.0, 1.0, 0.00]
    population["ξ"] = 0.01
    population["σ"] = 2.5
    
    return population
end

function create_default_npiparameters()
    npiparams_dict = Dict()
    ## It's important that the default parameters are those of the absence of 
    ## lockdowns, because they are the one the code refers to if the key "are_there_npi" = false
    npiparams_dict["κ₀s"] = [0.0]
    npiparams_dict["ϕs"] = [1.0]
    npiparams_dict["δs"] = [0.0]
    npiparams_dict["tᶜs"] =  [1]

    return npiparams_dict
end

function update_config!(config, cmd_line_args)
    # Define dictionary containing epidemic parameters
    if !haskey(config, "epidemic_params")
        config["epidemic_params"] = create_default_epi_params()
    end

    if !haskey(config, "population_params")
        config["population_params"] = create_default_population_params()
    end

    # Define dictionary containing npi parameters
    if (!haskey(config, "NPI") | !config["NPI"]["are_there_npi"] )
        config["NPI"] = create_default_npiparameters()
    end

    # overwrite config with command line
    if cmd_line_args["start-date"] !== nothing
        config["simulation"]["first_day_simulation"] = cmd_line_args["start-date"]
    end
    if cmd_line_args["end-date"] !== nothing
        config["simulation"]["last_day_simulation"] = cmd_line_args["end-date"]
    end
    if cmd_line_args["export-compartments-time-t"] !== nothing
        config["simulation"]["export_compartments_time_t"] = cmd_line_args["export-compartments-time-t"]
    end
    if cmd_line_args["export-compartments-full"] == true
        config["simulation"]["export_compartments_full"] = true
    end

    if cmd_line_args["initial-conditions"] !== nothing
        config["data"]["initial-conditions"] = cmd_line_args["initial-conditions"]
    end

    nothing
end

function init_pop_param_struct(G::Int64, M::Int64,
    G_coords::Array{String, 1},
    pop_params_dict::Dict, 
    metapop_df::DataFrame,
    network_df::DataFrame)

    # Subpopulations' patch surface
    sᵢ = metapop_df[:, "area"]
    # Subpopulation by age strata
    nᵢᵍ = copy(transpose(Array{Float64,2}(metapop_df[:, G_coords])))
    # Age Contact Matrix
    C = Float64.(mapreduce(permutedims, vcat, pop_params_dict["C"]))
    # Average number of contacts per strata
    kᵍ = Float64.(pop_params_dict["kᵍ"])
    # Average number of contacts at home per strata
    kᵍ_h = Float64.(pop_params_dict["kᵍ_h"])
    # Average number of contacts at work per strata
    kᵍ_w = Float64.(pop_params_dict["kᵍ_w"])
    # Degree of mobility per strata
    pᵍ = Float64.(pop_params_dict["pᵍ"])
    # Density factor
    ξ = pop_params_dict["σ"]
    # Average household size
    σ = pop_params_dict["σ"]

    edgelist = Array{Int64, 2}(network_df[:, 1:2])
    Rᵢⱼ      = copy(network_df[:, 3])
    edgelist, Rᵢⱼ = correct_self_loops(edgelist, Rᵢⱼ, M)

    return Population_Params(G, M, nᵢᵍ, kᵍ, kᵍ_h, kᵍ_w, C, pᵍ, edgelist, Rᵢⱼ, sᵢ, ξ, σ)
end

function init_epi_parameters_struct(G::Int64, M::Int64, T::Int64,
    G_coords::Array{String, 1}, 
    epi_params_dict::Dict)

    # Scaling of the asymptomatic infectivity
    scale_β = epi_params_dict["scale_β"]
    # Infectivity of Symptomatic
    βᴵ = epi_params_dict["βᴵ"]
    # Infectivity of Asymptomatic
    if haskey(epi_params_dict, "βᴬ")
    βᴬ = epi_params_dict["βᴬ"]
    elseif haskey(epi_params_dict, "scale_β")
    βᴬ = scale_β * βᴵ
    else
    @error "Either βᴬ or scale_β should be provided"
    end
    # Exposed rate
    ηᵍ = Float64.(epi_params_dict["ηᵍ"])
    # Asymptomatic rate
    αᵍ = Float64.(epi_params_dict["αᵍ"])
    # Infectious rate
    μᵍ = Float64.(epi_params_dict["μᵍ"])


    ## EPIDEMIC PARAMETERS TRANSITION RATES VACCINATION

    # Direct death probability
    θᵍ = Float64.(reduce(hcat, [epi_params_dict["θᵍ"], epi_params_dict["θᵍ"] * epi_params_dict["risk_reduction_dd"]]) )
    # Hospitalization probability
    γᵍ = Float64.(reduce(hcat, [epi_params_dict["γᵍ"], epi_params_dict["γᵍ"] * epi_params_dict["risk_reduction_h"]]) )
    # Fatality probability in ICU
    ωᵍ = Float64.(reduce(hcat, [epi_params_dict["ωᵍ"], epi_params_dict["ωᵍ"] * epi_params_dict["risk_reduction_d"]]) )
    # Pre-deceased rate
    ζᵍ = Float64.(epi_params_dict["ζᵍ"])
    # Pre-hospitalized in ICU rate
    λᵍ = Float64.(epi_params_dict["λᵍ"])
    # Death rate in ICU
    ψᵍ = Float64.(epi_params_dict["ψᵍ"])
    # ICU discharge rate
    χᵍ = Float64.(epi_params_dict["χᵍ"])
    

    return Epidemic_Params(βᴵ, βᴬ, ηᵍ, αᵍ, μᵍ, θᵍ, γᵍ, ζᵍ, λᵍ, ωᵍ, ψᵍ, χᵍ, G, M, T)
end

"""
save_simulation_hdf5(epi_params::Epidemic_Params,
                         population::Population_Params,
                         output_fname::String;
                         export_time_t = -1)

    Save the full simulations.

    # Arguments

    - `epi_params::Epidemic_Params`: Structure that contains all epidemic parameters
    and the epidemic spreading information.
    - `population::Population_Params`: Structure that contains all the parameters
    related with the population.
    - `output_fname::String`: Output filename.

    ## Optional

    - `export_time_t = -1`: Time step to ve saved instead of the full simulation.
"""
function save_simulation_hdf5(epi_params::Epidemic_Params, 
                              population::Population_Params,
                              output_fname;
                              export_time_t = -1)

    G = population.G
    M = population.M
    T = epi_params.T
    N = epi_params.NumComps

    compartments = zeros(Float64, G, M, T, V, N);
    compartments[:, :, :, 1]  .= epi_params.ρˢᵍ .* population.nᵢᵍ
    compartments[:, :, :, 2]  .= epi_params.ρᴱᵍ .* population.nᵢᵍ
    compartments[:, :, :, 3]  .= epi_params.ρᴬᵍ .* population.nᵢᵍ
    compartments[:, :, :, 4]  .= epi_params.ρᴵᵍ .* population.nᵢᵍ
    compartments[:, :, :, 5]  .= epi_params.ρᴾᴴᵍ .* population.nᵢᵍ
    compartments[:, :, :, 6]  .= epi_params.ρᴾᴰᵍ .* population.nᵢᵍ
    compartments[:, :, :, 7]  .= epi_params.ρᴴᴿᵍ .* population.nᵢᵍ
    compartments[:, :, :, 8]  .= epi_params.ρᴴᴰᵍ .* population.nᵢᵍ
    compartments[:, :, :, 9]  .= epi_params.ρᴿᵍ .* population.nᵢᵍ
    compartments[:, :, :, 10] .= epi_params.ρᴰᵍ .* population.nᵢᵍ
    if export_time_t > 0
        h5open(output_fname, "w") do file
            write(file, "data", compartments[:,:,export_time_t,:])
        end
    else
        h5open(output_fname, "w") do file
            write(file, "data", compartments[:,:,:,:])
        end
    end
end

"""
save_simulation_netCDF(epi_params::Epidemic_Params, 
                                    population::Population_Params,
                                    output_fname::String;
                                    G_coords= nothing,
                                    M_coords = nothing,
                                    T_coords = nothing
                                    )

    Save the full simulations.

    # Arguments

    - `epi_params::Epidemic_Params`: Structure that contains all epidemic parameters
    and the epidemic spreading information.
    - `population::Population_Params`: Structure that contains all the parameters
    related with the population.
    - `output_fname::String`: Output filename.

    ## Optional
    - `G_coords = nothing`: Array::{String} of size G containing the labels for age strata
    - `M_coords = nothing`: Array::{String} of size M containing the labels for the patches
    - `T_coords = nothing`: Array::{String} of size t containing the labels for the time (dates)
    - `export_time_t = -1`: Time step to ve saved instead of the full simulation.
"""
function save_simulation_netCDF( epi_params::Epidemic_Params, 
                                 population::Population_Params,
                                 output_fname::String;
                                 G_coords = nothing,
                                 M_coords = nothing,
                                 T_coords = nothing
                                )
    G = population.G
    M = population.M
    T = epi_params.T
    S = epi_params.NumComps
    comp_coords = epi_params.CompLabels

    if isnothing(G_coords)
        G_coords = collect(1:G)
    end
    if isnothing(M_coords)
        M_coords = collect(1:M)
    end
    if isnothing(T_coords)
        T_coords = collect(1:T) 
    end


    compartments = zeros(Float64, G, M, T, S);
    compartments[:, :, :, 1]  .= epi_params.ρˢᵍ .* population.nᵢᵍ
    compartments[:, :, :, 2]  .= epi_params.ρᴱᵍ .* population.nᵢᵍ
    compartments[:, :, :, 3]  .= epi_params.ρᴬᵍ .* population.nᵢᵍ
    compartments[:, :, :, 4]  .= epi_params.ρᴵᵍ .* population.nᵢᵍ
    compartments[:, :, :, 5]  .= epi_params.ρᴾᴴᵍ .* population.nᵢᵍ
    compartments[:, :, :, 6]  .= epi_params.ρᴾᴰᵍ .* population.nᵢᵍ
    compartments[:, :, :, 7]  .= epi_params.ρᴴᴿᵍ .* population.nᵢᵍ
    compartments[:, :, :, 8]  .= epi_params.ρᴴᴰᵍ .* population.nᵢᵍ
    compartments[:, :, :, 9]  .= epi_params.ρᴿᵍ .* population.nᵢᵍ
    compartments[:, :, :, 10] .= epi_params.ρᴰᵍ .* population.nᵢᵍ
    isfile(output_fname) && rm(output_fname)

    nccreate(output_fname, "data", "G", G_coords, "M", M_coords, "T", T_coords, "epi_states", collect(comp_coords))
    ncwrite(compartments, output_fname, "data")

end