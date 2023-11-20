using Plots
using EnhancedBayesianNetworks

cloudy = DiscreteRootNode(:C, Dict(:cloudy => 0.5, :sunny => 0.5))

random = ContinuousRootNode(:Rnd, Normal(), ExactDiscretization([-0.5, 0, 0.5]))

sprinkler_name = :SP
sprinkler_parents = [cloudy]
sprinkler_states = Dict(
    [:cloudy] => Dict(:on => 0.5, :off => 0.5),
    [:sunny] => Dict(:on => 0.9, :off => 0.1),
)
sprinkler_parameter = Dict(:on => [Parameter(-1, :sp)], :off => [Parameter(1, :sp)])

sprinkler_node = DiscreteChildNode(sprinkler_name, sprinkler_parents, sprinkler_states, sprinkler_parameter)

rain_name = :R
rain_parents = [cloudy]
rain_states = Dict(
    [:cloudy] => Dict(:rain => 0.8, :not_rain => 0.2),
    [:sunny] => Dict(:rain => 0.2, :not_rain => 0.8),
)
rain_parameter = Dict(:rain => [Parameter(-1, :rn)], :not_rain => [Parameter(1, :rn)])

rain_node = DiscreteChildNode(rain_name, rain_parents, rain_states, rain_parameter)

wetgrass_name = :WG

wetgrass_parents = [random, rain_node, sprinkler_node]
wetgrass_model1 = Model(df -> df.Rnd .^ 2 .* df.rn .+ df.sp, :final_state)
wetgrass_models = [wetgrass_model1]
wetgrass_simulations = MonteCarlo(200)
wetgrass_performances = df -> df.final_state
wetgrass_node = DiscreteFunctionalNode(wetgrass_name, wetgrass_parents, wetgrass_models, wetgrass_performances, wetgrass_simulations)

wetfloor_name = :WF
wetfloor_parents = [random, rain_node]
wetfloor_model1 = Model(df -> df.Rnd .^ 2 .* df.rn, :final_floor)
wetfloor_models = [wetfloor_model1]
wetfloor_simulations = MonteCarlo(200)

wetfloor_node = ContinuousFunctionalNode(wetfloor_name, wetfloor_parents, wetfloor_models, wetfloor_simulations)

nodes = [random, cloudy, rain_node, sprinkler_node, wetgrass_node, wetfloor_node]

ebn = EnhancedBayesianNetwork(nodes)

gr();
nodesize = 0.12
fontsize = 18
EnhancedBayesianNetworks.plot(ebn, :tree, nodesize, fontsize)
Plots.savefig("C:\\Users\\Administrator\\Resilio Sync\\PhD\\3_Academic\\Papers_Presentations\\Conferences\\2024_ICVRAM\\ASCE-ICVRAM-ISUMA 2024-Abstract-LaTeX\\Figures\\ebn.png")


# rbn = reduce!(ebn)
# EnhancedBayesianNetworks.plot(rbn, :tree, nodesize, fontsize)
# Plots.savefig("C:\\Users\\Administrator\\Resilio Sync\\PhD\\3_Academic\\Papers_Presentations\\Conferences\\2024_ICVRAM\\ASCE-ICVRAM-ISUMA 2024-Abstract-LaTeX\\Figures\\ebn.png")


# a = evaluate!(ebn)
# EnhancedBayesianNetworks.plot(a, :tree, nodesize, fontsize)
# Plots.savefig("C:\\Users\\Administrator\\Resilio Sync\\PhD\\3_Academic\\Papers_Presentations\\Conferences\\2024_ICVRAM\\ASCE-ICVRAM-ISUMA 2024-Abstract-LaTeX\\Figures\\rbn.png")
