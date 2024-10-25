using EnhancedBayesianNetworks

# tampering = DiscreteRootNode(:Tampering, Dict(:NoT => [0.98999, 0.99111], :YesT => [0.00889, 0.01001]))
# fire = DiscreteRootNode(:Fire, Dict(:NoF => [0.958978, 0.959989], :YesF => [0.00011, 0.041002]))

# alarm_states = Dict(
#     [:NoT, :NoF] => Dict(:NoA => [0.999800, 0.999997], :YesA => [0.000003, 0.000200]),
#     [:NoT, :YesF] => Dict(:NoA => [0.010000, 0.012658], :YesA => [0.987342, 0.990000]),
#     [:YesT, :NoF] => Dict(:NoA => [0.100000, 0.119999], :YesA => [0.880001, 0.900000]),
#     [:YesT, :YesF] => Dict(:NoA => [0.400000, 0.435894], :YesA => [0.564106, 0.600000])
# )

# alarm = EnhancedBayesianNetworks.NewDiscreteChildNode(:Alarm, alarm_states)

# smoke_state = Dict(
#     [:NoF] => Dict(:NoS => [0.897531, 0.915557], :YesS => [0.010000, 0.102469]),
#     [:YesF] => Dict(:NoS => [0.090000, 0.110000], :YesS => [0.890000, 0.910000])
# )
# smoke = EnhancedBayesianNetworks.NewDiscreteChildNode(:Smoke, smoke_state)

# leaving_state = Dict(
#     [:NoA] => Dict(:NoL => [0.585577, 0.599999], :YesL => [0.400001, 0.414423]),
#     [:YesA] => Dict(:NoL => [0.100000, 0.129999], :YesL => [0.870001, 0.900000])
# )
# leaving = EnhancedBayesianNetworks.NewDiscreteChildNode(:Leaving, leaving_state)

# report_state = Dict(
#     [:NoL] => Dict(:NoR => [0.809988, 0.828899], :YesR => [0.171101, 0.190012]),
#     [:YesL] => Dict(:NoR => [0.240011, 0.250000], :YesR => [0.750000, 0.759989])
# )
# report = EnhancedBayesianNetworks.NewDiscreteChildNode(:Report, report_state)

# nodes = [fire, alarm, smoke, tampering, leaving, report]
# EnhancedBayesianNetworks.add_child!(net, :Tampering, :Alarm)
# EnhancedBayesianNetworks.add_child!(net, :Fire, :Smoke)
# EnhancedBayesianNetworks.add_child!(net, :Fire, :Alarm)
# EnhancedBayesianNetworks.add_child!(net, :Alarm, :Leaving)
# EnhancedBayesianNetworks.add_child!(net, :Leaving, :Report)
# EnhancedBayesianNetworks.order_net!(net)

M = 3.2633
m = 0.8158
g = 981
A = DiscreteRootNode(:A, Dict(:road => 0.7, :offroad => 0.3), Dict(:road => [Parameter(0.15915, :A)], :offroad => [Parameter(0.8, :A)]))
b₀ = DiscreteRootNode(:b₀, Dict(:normal_load => 0.7, :over_load => 0.3), Dict(:normal_load => [Parameter(0.27, :b₀)], :over_load => [Parameter(0.5, :b₀)]))
discretization_v = ExactDiscretization(collect(range(8.5, 11.5, 4)))
V = ContinuousRootNode(:V, Uniform(7, 12), discretization_v)

C = ContinuousRootNode(:C, Normal(431.7221, 10))
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
sim = MonteCarlo(2 * 10^6)

E = EnhancedBayesianNetworks.NewDiscreteFunctionalNode(:E, [model], performance, sim)

nodes = [A, b₀, V, C, Cₖ, K, E]
net = EnhancedBayesianNetworks.Network(nodes)

EnhancedBayesianNetworks.add_child!(net, :E, :A)
EnhancedBayesianNetworks.add_child!(net, :E, :b₀)
EnhancedBayesianNetworks.add_child!(net, :E, :V)
EnhancedBayesianNetworks.add_child!(net, :E, :C)
EnhancedBayesianNetworks.add_child!(net, :E, :Cₖ)
EnhancedBayesianNetworks.add_child!(net, :E, :K)

EnhancedBayesianNetworks.order_net!(net)