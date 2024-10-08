using EnhancedBayesianNetworks
using Plots
# using PGFPlotsX

root1 = DiscreteRootNode(:X1, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :X1)], :n => [Parameter(0, :X1)]))
root2 = DiscreteRootNode(:X2, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(2.2, :X2)], :no => [Parameter(5.5, :X2)]))
root3 = ContinuousRootNode(:Y1, Normal(), ExactDiscretization([0, 0.2, 1]))
root4 = ContinuousRootNode(:Y2, Normal())

functional1_parents = [root1, root3]
disc_D = ApproximatedDiscretization([-1.1, 0, 0.11], 2)
model1 = [Model(df -> (df.X1 .^ 2) ./ 2 .- df.Y1, :fun1)]
simulation1 = MonteCarlo(300)
functional1_node = ContinuousFunctionalNode(:F1, functional1_parents, model1, simulation1, disc_D)


standard3_parents = [root1]
standard3_states = Dict(
    [:y] => Normal(),
    [:n] => Normal(2, 2)
)
standard3_node = ContinuousChildNode(:cacca, standard3_parents, standard3_states)


functional2_name = :F2
functional2_parents = [root2, root3, root4, functional1_node]
model2 = [Model(df -> (df.X2 .^ 2 .+ df.Y1 .^ 2) ./ 2 .- df.Y1 .- df.Y2 .+ df.F1, :fun2)]
performance = df -> 1 .- 2 .* df.fun2
simulation2 = MonteCarlo(300)
functional2_node = DiscreteFunctionalNode(functional2_name, functional2_parents, model2, performance, simulation2)

## Already build everything with evidence.
nodes = [root1, root2, root3, root4, functional1_node, functional2_node, standard3_node]
ebn = EnhancedBayesianNetwork(nodes)

r_ebn = evaluate!(ebn)

# ##TODO creare nuovo esempio per avere i plot fatti bene per evidence e reduction
# gr();
# nodesize = 0.1
# fontsize = 18
# EnhancedBayesianNetworks.plot(ebn, :tree, nodesize, fontsize)
# Plots.savefig("/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Conferences/2023_ESREL/ExtendedAbstract-Template/imgs/ebn_without_evidence.png")

# EnhancedBayesianNetworks.plot(ebn_D, :tree, nodesize, fontsize)
# Plots.savefig("/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Conferences/2023_ESREL/ExtendedAbstract-Template/imgs/ebn_with_evidence.png")

# EnhancedBayesianNetworks.plot(rbn, :tree, nodesize, fontsize)
# Plots.savefig("/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Conferences/2023_ESREL/ExtendedAbstract-Template/imgs/reducedbn.png")



# rbn = a[1]
# bn = BayesianNetwork(rbn)
# query = [:x]
# e = Dict(:f1 => :f)
# inf = InferenceState(bn, query, e)


# infer(bn, query, e)
