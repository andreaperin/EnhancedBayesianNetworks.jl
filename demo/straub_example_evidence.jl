using EnhancedBayesianNetworks
using Distributions
# using Plots
using .MathConstants: γ
using UncertaintyQuantification

Uᵣ = ContinuousRootNode(:Uᵣ, Normal())
μ_gamma = 60
cov_gamma = 0.2
α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
V = ContinuousRootNode(:V, Gamma(α, θ))

μ_gumbel = 50
cov_gumbel = 0.4
μ_loc, β = distribution_parameters(μ_gumbel, cov_gumbel * μ_gumbel, Gumbel)

H = ContinuousRootNode(:H, Gumbel(μ_loc, β))

n = 100000

function plastic_moment_capacities(uᵣ)
    ρ = 0.5477
    μ = 150
    cov = 0.2

    λ, ζ = distribution_parameters(μ, μ * cov, LogNormal)

    normal_μ = λ + ρ * ζ * uᵣ
    normal_std = sqrt((1 - ρ^2) * ζ^2)
    exp(rand(Normal(normal_μ, normal_std), 1)[1])
end

parents = [Uᵣ]
model1 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r1)
model2 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r2)
model3 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r3)
model4 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r4)
model5 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r5)

discretization1 = ApproximatedDiscretization(collect(range(50, 250, 10)), 1)
discretization2 = ApproximatedDiscretization(collect(range(50.01, 250.01, 10)), 1)

R1 = ContinuousFunctionalNode(:R1, parents, [model1], MonteCarlo(n))
R2 = ContinuousFunctionalNode(:R2, parents, [model2], MonteCarlo(n))
R3 = ContinuousFunctionalNode(:R3, parents, [model3], MonteCarlo(n))
R4 = ContinuousFunctionalNode(:R4, parents, [model4], MonteCarlo(n), discretization1)
R5 = ContinuousFunctionalNode(:R5, parents, [model5], MonteCarlo(n), discretization2)


function frame_model(r1, r2, r3, r4, r5, v, h)
    g1 = r1 + r2 + r4 + r5 - 5 * h
    g2 = r2 + 2 * r3 + r4 - 5 * v
    g3 = r1 + 2 * r3 + 2 * r4 + r5 - 5 * h - 5 * v
    return minimum([g1, g2, g3])
end

model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.R4, df.R5, df.V, df.H), :G)
performance = df -> df.G
simulation = MonteCarlo(n)
frame = DiscreteFunctionalNode(:E, [R1, R2, R3, R4, R5, V, H], [model], performance, simulation)


nodes = [Uᵣ, V, H, R1, R2, R3, R4, R5, frame]
ebn = EnhancedBayesianNetwork(nodes)

eebn = EnhancedBayesianNetworks.evaluate(ebn)

eebn.nodes[end].states

evidence1 = Dict(
    :R5_d => Symbol([94.45444444444445, 116.67666666666666]),
    :R4_d => Symbol([63.72984058368004, 72.22222222222223])
)

ϕ1 = infer(eebn, :E, evidence1)

evidence2 = Dict(
    :R5_d => Symbol([94.45444444444445, 116.67666666666666]),
    :R4_d => Symbol([138.88888888888889, 161.11111111111111])
)

ϕ2 = infer(eebn, :E, evidence2)

evidence3 = Dict(
    :R5_d => Symbol([183.34333333333333, 205.56555555555556]),
    :R4_d => Symbol([138.88888888888889, 161.11111111111111])
)

ϕ3 = infer(eebn, :E, evidence3)




#### Checking with srps only


