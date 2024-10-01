using EnhancedBayesianNetworks
# using Plots
# ### ROOT
# interval = (1.10, 1.30)

# root1 = DiscreteRootNode(:A, Dict(:a1 => 0.5, :a2 => 0.5), Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))
# root2 = ContinuousRootNode(:B, interval)
# root3 = ContinuousRootNode(:P, Uniform(-10, 10))
# model = Model(df -> df.A .+ df.B .+ df.P, :C)
# sim = MonteCarlo(100_000)
# performance = df -> 2 .- df.C
# disc_functional = DiscreteFunctionalNode(:C, [root1, root2, root3], [model], performance, sim)

# nodes = [root1, root2, root3, disc_functional]
# ebn = EnhancedBayesianNetwork(nodes)
# # EnhancedBayesianNetworks.plot(ebn)
# na = EnhancedBayesianNetworks._evaluate(disc_functional)
# # n = EnhancedBayesianNetworks._evaluate(disc_functional)

### CHILD

using EnhancedBayesianNetworks
root1 = DiscreteRootNode(:A, Dict(:a1 => 0.5, :a2 => 0.5), Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))
root2 = ContinuousRootNode(:B, Normal())
root3 = DiscreteRootNode(:D, Dict(:d1 => 0.5, :d2 => 0.5), Dict(:d1 => [Parameter(1, :D)], :d2 => [Parameter(2, :D)]))
states = Dict(
    [:a1] => (0.1, 0.3),
    [:a2] => (0.7, 0.8)
)
child = ContinuousChildNode(:C1, [root1], states)

model = Model(df -> df.D .+ df.C1 .+ df.B, :C2)
sim = MonteCarlo(100_000)
performance = df -> 2 .- df.C2
parents = [root3, root2, child]
model_node = DiscreteFunctionalNode(:F1, parents, [model], performance, sim)

nodes = [root1, root2, root3, child, model_node]

ebn = EnhancedBayesianNetwork(nodes)
na = EnhancedBayesianNetworks._evaluate(model_node)