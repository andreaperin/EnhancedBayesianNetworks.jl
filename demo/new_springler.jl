using Plots
using EnhancedBayesianNetworks

cloudy = DiscreteRootNode(:C, Dict(:cloudy => 0.5, :sunny => 0.5))

random = ContinuousRootNode(:Rnd, Normal(), ExactDiscretization([-0.5, 0, 0.5]))

sprinkler_name = :SP
sprinkler_parents = Vector{AbstractNode}()
append!(sprinkler_parents, [cloudy])
sprinkler_states = Dict(
    [:cloudy] => Dict(:on => 0.5, :off => 0.5),
    [:sunny] => Dict(:on => 0.9, :off => 0.1),
)
sprinkler_parameter = Dict(:on => [Parameter(-1, :sp)], :off => [Parameter(1, :sp)])

sprinkler_node = DiscreteChildNode(sprinkler_name, sprinkler_parents, sprinkler_states, sprinkler_parameter)

rain_name = :R
rain_parents = Vector{AbstractNode}()
append!(rain_parents, [cloudy])
rain_states = Dict(
    [:cloudy] => Dict(:rain => 0.8, :not_rain => 0.2),
    [:sunny] => Dict(:rain => 0.2, :not_rain => 0.8),
)
rain_parameter = Dict(:rain => [Parameter(-1, :rn)], :not_rain => [Parameter(1, :rn)])

rain_node = DiscreteChildNode(rain_name, rain_parents, rain_states, rain_parameter)

wetgrass_name = :WG
wetgrass_parents = Vector{AbstractNode}()
append!(wetgrass_parents, [random, rain_node, sprinkler_node])
wetgrass_model1 = Model(df -> df.Rnd .^ 2 .* df.rn .+ df.sp, :final_state)
wetgrass_models = [wetgrass_model1]
wetgrass_simulation = MonteCarlo(200)
wetgrass_performances = df -> df.final_state
wetgrass_parameters = Dict(:fail_WG => [Parameter(1.0, :wgp)], :safe_WG => [Parameter(1.0, :wgp)])
wetgrass_node = DiscreteFunctionalNode(wetgrass_name, wetgrass_parents, wetgrass_models, wetgrass_performances, wetgrass_simulation, wetgrass_parameters)

wetgrass2_name = :WG2
wetgrass2_parents = Vector{AbstractNode}()
append!(wetgrass2_parents, [wetgrass_node, random])
wetgrass2_model1 = Model(df -> df.wgp .* df.Rnd, :final_state2)
wetgrass2_models = [wetgrass2_model1]
wetgrass2_simulation = MonteCarlo(200)
wetgrass2_models = [wetgrass2_model1]
wetgrass2_discretization = ApproximatedDiscretization([-1.1, 0, 0.11], 2)
wetgrass2_node = ContinuousFunctionalNode(wetgrass2_name, wetgrass2_parents, wetgrass2_models, wetgrass2_simulation, wetgrass2_discretization)


wetgrass3_name = :WG3
wetgrass3_parents = Vector{AbstractNode}()
append!(wetgrass3_parents, [wetgrass2_node])
wetgrass3_model1 = Model(df -> df.WG2, :final_state3)
wetgrass3_models = [wetgrass3_model1]
wetgrass3_simulation = MonteCarlo(200)
wetgrass3_performances = df -> df.final_state3
wetgrass3_node = DiscreteFunctionalNode(wetgrass3_name, wetgrass3_parents, wetgrass3_models, wetgrass3_performances, wetgrass3_simulation)


wetfloor_name = :WF
wetfloor_parents = Vector{AbstractNode}()
append!(wetfloor_parents, [random, rain_node])
wetfloor_model1 = Model(df -> df.Rnd .^ 2 .* df.rn, :final_floor)
wetfloor_models = [wetfloor_model1]
wetfloor_simulation = MonteCarlo(200)
wetfloor_node = ContinuousFunctionalNode(wetfloor_name, wetfloor_parents, wetfloor_models, wetfloor_simulation)

wetfloor_d_name = :dWF
wetfloor_d_parents = Vector{AbstractNode}()
append!(wetfloor_d_parents, [wetfloor_node])
wetfloor_d_model1 = Model(df -> df.final_floor .+ 1, :final_floor_d)
wetfloor_d_models = [wetfloor_d_model1]
wetfloor_d_simulation = MonteCarlo(200)
wetgfloor_d_performances = df -> df.final_floor_d .- 1
wetfloor_d_node = DiscreteFunctionalNode(wetfloor_d_name, wetfloor_d_parents, wetfloor_d_models, wetgfloor_d_performances, wetfloor_d_simulation)

nodes = [random, cloudy, rain_node, sprinkler_node, wetgrass_node, wetgrass2_node, wetgrass3_node, wetfloor_node, wetfloor_d_node]

ebn = EnhancedBayesianNetwork(nodes)



# oo = EnhancedBayesianNetworks.transfer_continuous(ebn)
# e_ebn, b = EnhancedBayesianNetworks._evaluate_single_layer(oo)
# e_ebn2, b2 = EnhancedBayesianNetworks._evaluate_single_layer(e_ebn)

# e_ebn2, b2 = EnhancedBayesianNetworks._evaluate_single_layer(e_ebn2)

ooo = evaluate(ebn)
# EnhancedBayesianNetworks.plot(ooo)
# model_nodes=filter(x->x.name ∈ [:WG, :WG3, :dWF], ooo.nodes)
# rr = evaluate!(oo)


# gr();
# nodesize = 0.12
# fontsize = 18
# EnhancedBayesianNetworks.plot(ebn, :tree, nodesize, fontsize)
# Plots.savefig("C:\\Users\\Administrator\\Resilio Sync\\PhD\\3_Academic\\Papers_Presentations\\Conferences\\2024_ICVRAM\\ASCE-ICVRAM-ISUMA 2024-Abstract-LaTeX\\Figures\\ebn.png")


# rbn = reduce!(ebn)
# EnhancedBayesianNetworks.plot(rbn, :tree, nodesize, fontsize)
# Plots.savefig("C:\\Users\\Administrator\\Resilio Sync\\PhD\\3_Academic\\Papers_Presentations\\Conferences\\2024_ICVRAM\\ASCE-ICVRAM-ISUMA 2024-Abstract-LaTeX\\Figures\\ebn.png")


# a = evaluate!(ebn)
# EnhancedBayesianNetworks.plot(a, :tree, nodesize, fontsize)
# Plots.savefig("C:\\Users\\Administrator\\Resilio Sync\\PhD\\3_Academic\\Papers_Presentations\\Conferences\\2024_ICVRAM\\ASCE-ICVRAM-ISUMA 2024-Abstract-LaTeX\\Figures\\rbn.png")


