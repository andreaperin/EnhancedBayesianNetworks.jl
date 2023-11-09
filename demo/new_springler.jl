using Plots
using EnhancedBayesianNetworks

cloudy = DiscreteRootNode(:cloudy, Dict(:cloudy => 0.5, :sunny => 0.5))

random = ContinuousRootNode(:random, Normal(), ExactDiscretization([-0.5, 0, 0.5]))

sprinkler_name = :sprinkler
sprinkler_parents = [cloudy]
sprinkler_states = Dict(
    [:cloudy] => Dict(:on => 0.5, :off => 0.5),
    [:sunny] => Dict(:on => 0.9, :off => 0.1),
)
sprinkler_parameter = Dict(:on => [Parameter(-1, :sp)], :off => [Parameter(1, :sp)])

sprinkler_node = DiscreteChildNode(sprinkler_name, sprinkler_parents, sprinkler_states, sprinkler_parameter)

rain_name = :rain
rain_parents = [cloudy]
rain_states = Dict(
    [:cloudy] => Dict(:rain => 0.8, :not_rain => 0.2),
    [:sunny] => Dict(:rain => 0.2, :not_rain => 0.8),
)
rain_parameter = Dict(:rain => [Parameter(-1, :rn)], :not_rain => [Parameter(1, :rn)])

rain_node = DiscreteChildNode(rain_name, rain_parents, rain_states, rain_parameter)

wetgrass_name = :wetgrass

wetgrass_parents = [random, rain_node, sprinkler_node]
wetgrass_model1 = Model(df -> df.random .^ 2 .* df.rn .+ df.sp, :final_state)
wetgrass_models = [wetgrass_model1]
wetgrass_simulations = MonteCarlo(200)
wetgrass_performances = df -> df.final_state
wetgrass_node = DiscreteFunctionalNode(wetgrass_name, wetgrass_parents, wetgrass_models, wetgrass_performances, wetgrass_simulations)

wetfloor_name = :wetfloor
wetfloor_parents = [random, rain_node]
wetfloor_model1 = Model(df -> df.random .^ 2 .* df.rn, :final_floor)
wetfloor_models = [wetfloor_model1]
wetfloor_simulations = MonteCarlo(200)

wetfloor_node = ContinuousFunctionalNode(wetfloor_name, wetfloor_parents, wetfloor_models, wetfloor_simulations)

nodes = [random, cloudy, rain_node, sprinkler_node, wetgrass_node, wetfloor_node]

ebn = EnhancedBayesianNetwork(nodes)

a = evaluate!(ebn)
