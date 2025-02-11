using EnhancedBayesianNetworks
using .MathConstants: γ

n = 10^6
Uᵣ = ContinuousNode{UnivariateDistribution}(:Uᵣ, DataFrame(:Π => Normal()))
μ_gamma = 60
cov_gamma = 0.2
α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
V = ContinuousNode{UnivariateDistribution}(:V, DataFrame(:Π => Gamma(α, θ)))

μ_gumbel = 50
cov_gumbel = 0.4
μ_loc, β = distribution_parameters(μ_gumbel, cov_gumbel * μ_gumbel, Gumbel)
H = ContinuousNode{UnivariateDistribution}(:H, DataFrame(:Π => Gumbel(μ_loc, β)))

function plastic_moment_capacities(uᵣ)
    ρ = 0.5477
    μ = 150
    cov = 0.2

    λ, ζ = distribution_parameters(μ, μ * cov, LogNormal)

    normal_μ = λ + ρ * ζ * uᵣ
    normal_std = sqrt((1 - ρ^2) * ζ^2)
    exp(rand(Normal(normal_μ, normal_std)))
end

model1 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r1)
model2 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r2)
model3 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r3)
model4 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r4)
model5 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r5)

function frame_model(r1, r2, r3, r4, r5, v, h)
    g1 = r1 + r2 + r4 + r5 - 5 * h
    g2 = r2 + 2 * r3 + r4 - 5 * v
    g3 = r1 + 2 * r3 + 2 * r4 + r5 - 5 * h - 5 * v
    return minimum([g1, g2, g3])
end

R1 = ContinuousFunctionalNode(:R1, [model1], MonteCarlo(n))
R2 = ContinuousFunctionalNode(:R2, [model2], MonteCarlo(n))
R3 = ContinuousFunctionalNode(:R3, [model3], MonteCarlo(n))
R4 = ContinuousFunctionalNode(:R4, [model4], MonteCarlo(n))
R5 = ContinuousFunctionalNode(:R5, [model5], MonteCarlo(n))

model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.r4, df.r5, df.V, df.H), :G)
performance = df -> df.G
simulation = MonteCarlo(n)
frame = DiscreteFunctionalNode(:E, [model], performance, simulation)

nodes = [Uᵣ, V, H, R1, R2, R3, R4, R5, frame]

net = EnhancedBayesianNetwork(nodes)

add_child!(net, Uᵣ, R1)
add_child!(net, Uᵣ, R2)
add_child!(net, Uᵣ, R3)
add_child!(net, Uᵣ, R4)
add_child!(net, Uᵣ, R5)
add_child!(net, R1, frame)
add_child!(net, R2, frame)
add_child!(net, R3, frame)
add_child!(net, R4, frame)
add_child!(net, R5, frame)
add_child!(net, V, frame)
add_child!(net, H, frame)
order!(net)
evaluate!(net)

all(isapprox.(net.nodes[end].cpt[!, :Π], [0.026129, 0.973871]; atol=0.01))
