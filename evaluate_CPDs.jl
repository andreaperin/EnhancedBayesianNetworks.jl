using GraphPlot
include("CPDs.jl")
include("nodes.jl")
include("buildmodel_TH.jl")
include("models_probabilities.jl")
include("bn.jl")

a = NamedCategorical([:first, :second, :third], [1.34, 1.33, 1.33])
CPDa = StaticCPD(:time_scenario, a)
timescenario = StdNode(CPDa)

c1 = NamedCategorical([:low, :medium, :high], [0.5, 0.3, 0.2])
CPDc = StaticCPD(:grandparent, c1)
grandparent = StdNode(CPDc)

parents_dispersivitivy_longv = [timescenario, grandparent]
dispersivitivy_longv1 = truncated(Normal(1, 1), lower=0)
dispersivitivy_longv2 = truncated(Normal(2, 1), lower=0)
dispersivitivy_longv3 = truncated(Normal(2, 2), lower=0.1)
dispersivitivy_longv4 = truncated(Normal(10, 1), lower=0)
dispersivitivy_longv5 = truncated(Normal(20, 1), lower=0)
dispersivitivy_longv6 = truncated(Normal(22, 2), lower=0.1)
CPDd = CategoricalCPD{Distribution}(:dispersivity, [:time_scenario, :grandparent], [3, 3],
    [dispersivitivy_longv1,
        dispersivitivy_longv2,
        dispersivitivy_longv3,
        dispersivitivy_longv4,
        dispersivitivy_longv5,
        dispersivitivy_longv6]
)
dispersivitivy_longv = StdNode(CPDd, parents_dispersivitivy_longv)

flow1 = NamedCategorical([:small, :notsmall], [0.1, 0.9])
flow2 = NamedCategorical([:small, :notsmall], [0.2, 0.8])
flow3 = NamedCategorical([:small, :notsmall], [0.5, 0.5])
CPDb = CategoricalCPD(:pumpflowrate, [:time_scenario], [3],
    [flow1, flow2, flow3])
parents_b = [timescenario]
pumpflowrate_dict = Dict{Symbol,Dict{String,Vector}}(
    :small => Dict(
        "UQInputs" => [Parameter(-3e-5, :flow)],
        "FormatSpec" => [Dict(:flow => FormatSpec(".8e"))]
    ), :notsmall => Dict(
        "UQInputs" => [Parameter(-6e-5, :flow)],
        "FormatSpec" => [Dict(:flow => FormatSpec(".8e"))]
    )
)
pumpflowrate = StdNode(CPDb, parents_b, pumpflowrate_dict)


parents_Kz = [timescenario, grandparent]
kz1 = NamedCategorical([:t, :f], [0.1, 0.9])
kz2 = NamedCategorical([:t, :f], [0.2, 0.8])
kz3 = NamedCategorical([:t, :f], [0.5, 0.5])
kz4 = NamedCategorical([:t, :f], [0.1, 0.9])
kz5 = NamedCategorical([:t, :f], [0.2, 0.8])
kz6 = NamedCategorical([:t, :f], [0.5, 0.5])
kz7 = NamedCategorical([:t, :f], [0.1, 0.9])
kz8 = NamedCategorical([:t, :f], [0.2, 0.8])
kz9 = NamedCategorical([:t, :f], [0.5, 0.5])
CPDKz = CategoricalCPD(:Kz, [:time_scenario, :grandparent], [3, 3],
    [kz1, kz2, kz3, kz4, kz5, kz6, kz7, kz8, kz9])
kz_dict = Dict{Symbol,Dict{String,Vector}}(
    :t => Dict(
        "UQInputs" => [Parameter(0.00005, :Kz)],
        "FormatSpec" => [Dict(:Kz => FormatSpec(".8e"))]
    ), :f => Dict(
        "UQInputs" => [Parameter(0.0005, :Kz)],
        "FormatSpec" => [Dict(:Kz => FormatSpec(".8e"))]
    )
)
Kz = StdNode(CPDKz, parents_Kz, kz_dict)

######################################################################
#####                  simulation day node                      ######
######################################################################

day1 = NamedCategorical([:short, :mid, :long], [1.0, 0.0, 0.0])
day2 = NamedCategorical([:short, :mid, :long], [0.0, 1.0, 0.0])
day3 = NamedCategorical([:short, :mid, :long], [0.0, 0.0, 1.0])
CPDdays = CategoricalCPD(:simulation_days, [:time_scenario], [3],
    [day1, day2, day3])
simulation_days_dict = Dict{Symbol,Dict{String,Vector}}(
    :short => Dict(
        "UQInputs" => [Parameter(1, :sim_duration), Parameter(1, :time_interval)],
        "FormatSpec" => [Dict(:sim_duration => FormatSpec("d")), Dict(:time_interval => FormatSpec("d"))]
    ), :mid => Dict(
        "UQInputs" => [Parameter(10, :sim_duration), Parameter(10, :time_interval)],
        "FormatSpec" => [Dict(:sim_duration => FormatSpec("d")), Dict(:time_interval => FormatSpec("d"))]
    ), :long => Dict(
        "UQInputs" => [Parameter(100, :sim_duration), Parameter(100, :time_interval)],
        "FormatSpec" => [Dict(:sim_duration => FormatSpec("d")), Dict(:time_interval => FormatSpec("d"))]
    ),
)
parents_day = [timescenario]
simulation_days = StdNode(CPDdays, parents_day, simulation_days_dict)


######################################################################
#####                       Model node                          ######
######################################################################
system_model = string()
if Sys.iswindows()
    system_model = "model_TH_win"
elseif Sys.isapple()
    system_model = "model_TH_macos"
else
    @show("model must be compiled in this operating system")
end

model_input_folder = joinpath(system_model, "inputs")

sim_th = MonteCarlo(100)
parents_th = [simulation_days, pumpflowrate, Kz]
default_file = joinpath(model_input_folder, "default_th_values.xlsx")
default_inputs_th = get_default_inputs_and_format(default_file)
sourcedir = joinpath(pwd(), system_model)
source_file = "smoker.data"
extras = String[]
solvername = "smokerV3TC"
output_parameters = xlsx2output_parameter(default_file)
performances = build_performances(output_parameters)
th_node = ModelNode(:th_node, parents_th, default_inputs_th, sourcedir, source_file, extras, solvername, output_parameters, performances, true, sim_th)

inputs_mapping_dict, updated_inputs = get_inputs_mapping_dict1(th_node)
new_ordered_parents = get_new_ordered_parents(th_node)

# map_state_to_integer(updated_inputs, th_node)
th_node = ModelNode(:th_node, new_ordered_parents, default_inputs_th, sourcedir, source_file, extras, solvername, output_parameters, performances, true, inputs_mapping_dict, updated_inputs, sim_th)

prob, cpd, CPDth, th_node_final = evaluate_cpd_from_model(th_node, inputs_mapping_dict, performances, updated_inputs)

datetime = Dates.format(now(), "YYYY-mm-dd-HH-MM-SS")
path_to_store_cpds_table = joinpath(model_input_folder, "CPDs_table", datetime)
if ispath(path_to_store_cpds_table) == false
    mkpath(path_to_store_cpds_table)
end
parentsnames = join([string("_" * string(i)) for i in name.(th_node.parents)])

spec_string = string(typeof(th_node.sim)) * parentsnames
@save joinpath(path_to_store_cpds_table, "sim$(spec_string).jld2") prob

# f = jldopen(joinpath(path_to_store_cpds_table, "sim$(spec_string).jld2"), "r")
# cond_probs_dict = f["cond_probs_dict"]

nodes = [timescenario, grandparent, pumpflowrate, Kz, simulation_days, th_node_final, dispersivitivy_longv]
dag = _build_DiAGraph_from_nodes(nodes)
ordered_cpds, ordered_nodes, ordered_name_to_index, ordered_dag = _topological_ordered_dag(nodes)

th_bn = StdBayesNet(ordered_nodes)

## TODO Check with Jasper how to plot BN in the proper way

gplot(ordered_dag,
    nodelabel=name.(ordered_nodes),
    layout=stressmajorize_layout,
    nodefillc="lightgray",
    edgestrokec="black",
    EDGELINEWIDTH=0.3)

## TODO Check with Jasper Node Eliminatio Algo in MatLab [/Users/andreaperin_macos/Documents/PhD/3_Academic/Code/Matlab/OpenCossan/+opencossan/+bayesiannetworks]

