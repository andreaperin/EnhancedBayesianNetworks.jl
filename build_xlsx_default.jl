include("ebnprova.jl")
include("buildmodel_TH.jl")
include("models_probabilities.jl")

a = NamedCategorical([:first, :second, :third], [0.34, 0.33, 0.33])
CPDa = StaticCPD(:time_scenario, a)
timescenario = Node(CPDa)


b1 = NamedCategorical([:happen, :nothappen], [0.1, 0.9])
b2 = NamedCategorical([:happen, :nothappen], [0.2, 0.8])
b3 = NamedCategorical([:happen, :nothappen], [0.5, 0.5])
CPDb = CategoricalCPD(:earthquake, [:time_scenario], [3],
    [b1, b2, b3])
parents_b = [timescenario]

earthquake = Node(CPDb, parents_b)

c1 = NamedCategorical([:low, :medium, :high], [0.5, 0.3, 0.2])
c2 = NamedCategorical([:low, :medium, :high], [0.3, 0.5, 0.2])
c3 = NamedCategorical([:low, :medium, :high], [0.4, 0.2, 0.4])
CPDc = CategoricalCPD(:extremerain, [:time_scenario], [3],
    [c1, c2, c3])
parents_c = [timescenario]
extremerain = Node(CPDc, parents_c)

d1 = Normal(4, 1)
d2 = Normal(5, 3)
d3 = Uniform(1, 10)
CPDd = CategoricalCPD{Distribution}(:continous_node, [:time_scenario], [3], [d1, d2, d3])
parents_d = [timescenario]
continuous = Node(CPDd, parents_d)


parents_th = [timescenario, earthquake, extremerain, continuous]
th_node = NodeToBe(:th_node, parents_th)

states_df, states_comb = get_discreteparents_states_combinations(th_node)
inputs_mapping_vector = create_all_input_template(th_node, "model_TH/inputs", "default_th_values.xlsx")
