using UncertaintyQuantification
include("../src/nodes.jl")

emission = NamedCategorical([:nothappen, :happen], [0.5, 0.7])
CPD_emission = RootCPD(:emission, [emission])
emission_node = Node(CPD_emission)

h_distribution = Normal(0.24, 0.01)
CPD_h = RootCPD(:h, [h_distribution])
h_node = Node(CPD_h)

μ = log(10e9^2 / sqrt(1.6e9^2 + 10e9^2))
σ = sqrt(log(1.6e9^2 / 10e9^2 + 1))
E_distribution = LogNormal(μ, σ)
CPD_E = RootCPD(:E, [E_distribution])
E_node = Node(CPD_E)

μ = log(5000^2 / sqrt(400^2 + 5000^2))
σ = sqrt(log(400^2 / 5000^2 + 1))
P_distribution = LogNormal(μ, σ)
CPD_P = RootCPD(:P, [P_distribution])
P_node = Node(CPD_P)

μ = log(600^2 / sqrt(140^2 + 600^2))
σ = sqrt(log(140^2 / 600^2 + 1))
ρ_distribution = LogNormal(μ, σ)
CPD_ρ = RootCPD(:ρ, [ρ_distribution])
ρ_node = Node(CPD_ρ)
##TODO Continue from here implementing dependency between rvs as an argument of SRP states_dictionary

## Output Node FunctionalCPD
## Scenario1 (Emission_node = 1)

output_target = :output
output_parents = [emission_node, E_node, ρ_node, P_node, h_node]
output_parental_ncat = [2]


l1 = Parameter(1.8, :l) # length
b1 = Parameter(0.12, :b) # width
inertia = Model(df -> df.b .* df.h .^ 3 / 12, :I)
displacement = Model(
    df ->
        (df.ρ .* 9.81 .* df.b .* df.h .* df.l .^ 4) ./ (8 .* df.E .* df.I) .+
        (df.P .* df.l .^ 3) ./ (3 .* df.E .* df.I),
    :w,
)
model1 = [inertia, displacement]
parameter1 = [l1, b1]
max_displacement1 = 0.01
performance1 = df -> max_displacement1 .- df.w
correlated_nodes1 = name.([E_node, ρ_node])
copula1 = GaussianCopula([1 0.8; 0.8 1])
name1 = :jd
correlation1 = [CPDCorrelationCopula(correlated_nodes1, copula1, :jd)]
simulation1 = SubSetSimulation(2000, 0.1, 10, Uniform(-0.5, 0.5))
## simulation should be an argument of node elimination alghorithm only (not node)
structure1 = [model1, parameter1, performance1, correlation1]

output_target = :output
output_parents = [emission_node, E_node, ρ_node, P_node, h_node]
output_parental_ncat = [2]


l2 = Parameter(2.8, :l) # length
b2 = Parameter(1.12, :b) # width
inertia2 = Model(df -> df.b .* df.h .^ 3 / 12, :I)
displacement2 = Model(
    df ->
        (df.ρ .* 10.81 .* df.b .* df.h .* df.l .^ 4) ./ (8 .* df.E .* df.I) .+
        (df.P .* df.l .^ 3) ./ (3 .* df.E .* df.I),
    :w,
)
model2 = [inertia, displacement]
parameter2 = [l2, b2]
max_displacement2 = 0.05
performance2 = df -> max_displacement1 .- df.w
correlated_nodes2 = name.([E_node, ρ_node])
copula2 = GaussianCopula([1 0.8; 0.8 1])
name2 = :jd
correlation2 = [CPDCorrelationCopula(correlated_nodes2, copula2, :jd)]
simulation2 = SubSetSimulation(2000, 0.1, 10, Uniform(-0.5, 0.5))

structure2 = [model2, parameter2, performance2, correlation2, simulation2]

output_distributions = new_functional_cpd(:output, name.(output_parents), [2], [structure1, structure2])




nodes = [output_node, E_node, ρ_node, P_node, h_node, emission_node]
ebn = EnhancedBayesNet(nodes)
show(ebn)

""" Solving EnhancedBayesNet (No undefined continuous parents case) """
results = Dict()
for prob_dict in output_node.node_prob_dict
    UQInputs = vcat(build_UQInputs_singlecase(output_node, prob_dict), prob_dict.distribution.parameters)
    pf, cov, samples = probability_of_failure(prob_dict.distribution.model, prob_dict.distribution.performance, UQInputs, prob_dict.distribution.simulation)
    samples[!, :performance] = prob_dict.distribution.performance(samples)
    results[prob_dict] = Dict("pf" => pf, "cov" => cov, "samples" => samples)
end

## TODO try to perform probability_of_failure with jonas code building a specific post-processing function from matrix to 1 single value