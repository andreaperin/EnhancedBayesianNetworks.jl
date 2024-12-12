using EnhancedBayesianNetworks

M = 3.2633
m = 0.8158
g = 981
A = DiscreteRootNode(:A, Dict(:road => 0.7, :offroad => 0.3), Dict(:road => [Parameter(0.15915, :A)], :offroad => [Parameter(0.8, :A)]))
# b₀ = DiscreteRootNode(:b₀, Dict(:normal_load => [0.7, 0.9], :over_load => [0.1, 0.3]), Dict(:normal_load => [Parameter(0.27, :b₀)], :over_load => [Parameter(0.5, :b₀)]))
b₀ = DiscreteRootNode(:b₀, Dict(:normal_load => 0.7, :over_load => 0.3), Dict(:normal_load => [Parameter(0.27, :b₀)], :over_load => [Parameter(0.5, :b₀)]))

disc_v = ApproximatedDiscretization([7, 8, 10, 12], 1)
# v_states =
#     v_dist = Dict(
#         [:road] => UnamedProbabilityBox{Normal}([Interval(7, 8, :μ), Interval(1, 2, :σ)]),
#         [:offroad] => UnamedProbabilityBox{Uniform}([Interval(10, 11, :a), Interval(12, 13, :b)])
#     )
# v_states =
#     v_dist = Dict(
#         [:road] => (7, 8),
#         [:offroad] => (12, 13)
#     )
# V = ContinuousChildNode(:V, [A], v_states, disc_v)
# h, j = EnhancedBayesianNetworks._discretize(V)



discretization_v = ExactDiscretization(collect(range(8.5, 11.5, 4)))
# V = ContinuousRootNode(:V, Uniform(7, 12), discretization_v)
dist_v = (7, 13)
V = ContinuousRootNode(:V, dist_v, discretization_v)

# h, j = EnhancedBayesianNetworks._discretize(V)

# discretization_v2 = ExactDiscretization([-1, 0, 1])
# dist_v2 = UnamedProbabilityBox{Normal}([Interval(0, 1, :μ), Interval(1, 2, :σ)])
# V = ContinuousRootNode(:V, dist_v2, discretization_v2)

# C_dist = (400, 480)
C_dist = Normal(430, 10)
C = ContinuousRootNode(:C, C_dist)
Cₖ = ContinuousRootNode(:Cₖ, Normal(1475.5503, 10))
K = ContinuousRootNode(:K, Normal(55.0406, 10))

function composite_model(A, b₀, V, M, m, g, C, Cₖ, K)
    g1 = 1 .- (π .* m .* V .* A) ./ (b₀ .* K .* g .^ 2) .* [(Cₖ ./ (m .+ M) .- (C ./ M)) .^ 2 .+ C .^ 2 ./ (m .* M) .+ Cₖ .* K .^ 2 ./ (m .* M .^ 2)]
    g2 = 4000 .* C .* (M .* g) .^ (-1.5) .- 8.6394
    g3 = 2 .* .√(M .* g .* (K .^ 2 .* Cₖ ./ (C .* (m .+ M)) .+ C)) .- 1
    g4 = Cₖ .- [g .* (M .+ m)] .^ 0.877
    return minimum([g1[1], g2, g3, g4[1]])
end
model = Model(df -> composite_model.(df.A, df.b₀, df.V, M, m, g, df.C, df.Cₖ, df.K), :y)
performance = df -> df.y
# sim = DoubleLoop(MonteCarlo(100))
sim = MonteCarlo(18)
E = DiscreteFunctionalNode(:E, [A, b₀, V, C, Cₖ, K], [model], performance, sim)

nodes = [A, b₀, V, C, Cₖ, K, E]
ebn = EnhancedBayesianNetwork(nodes)

# h, j = EnhancedBayesianNetworks._discretize(V2)

# plt1 = EnhancedBayesianNetworks.plot(ebn, :spring, 0.2, 13)
# savefig(plt1, "/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Papers/elsarticle-template/imgs/fig8_example_vehicle.png")
eebn = evaluate(ebn)
plt2 = EnhancedBayesianNetworks.plot(eebn, :spring, 0.15, 13)
savefig(plt2, "/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Papers/elsarticle-template/imgs/fig9_example_vehicle_rbn.png")

eebn.nodes[end].states

evidence1 = Dict(
    :V_d => Symbol([9.5, 10.5]),
    :A => :road,
    :b₀ => :normal_load
)

ϕ1 = infer(eebn, :E, evidence1)

##  MonteCarlo Checking

M = Parameter(3.2633, :M)
m = Parameter(0.8158, :m)
g = Parameter(981, :g)
A = Parameter(0.15915, :A)
b₀ = Parameter(0.27, :b₀)
V = Parameter(10, :V)
C = RandomVariable(Normal(431.7221, 10), :C)
Cₖ = RandomVariable(Normal(1475.5503, 10), :Cₖ)
K = RandomVariable(Normal(55.0406, 10), :K)

model = Model(df -> composite_model.(df.A, df.b₀, df.V, df.M, df.m, df.g, df.C, df.Cₖ, df.K), :y)

inputs = [A, b₀, V, M, m, g, C, Cₖ, K]
sim = MonteCarlo(2 * 10^6)

pf = probability_of_failure(model, performance, inputs, sim)