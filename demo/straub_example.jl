using EnhancedBayesianNetworks
using Distributions
# using Plots
using .MathConstants: γ
# using UncertaintyQuantification

n = 10^6
Uᵣ = ContinuousRootNode(:Uᵣ, Normal())
μ_gamma = 60
cov_gamma = 0.2
α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
V = ContinuousRootNode(:V, Gamma(α, θ))

μ_gumbel = 50
cov_gumbel = 0.4
μ_loc, β = distribution_parameters(μ_gumbel, cov_gumbel * μ_gumbel, Gumbel)
H = ContinuousRootNode(:H, Gumbel(μ_loc, β))

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

n2 = 1000
discretization1 = ApproximatedDiscretization(collect(range(50, 250, 21)), 1)
discretization2 = ApproximatedDiscretization(collect(range(50.01, 250.01, 21)), 1)
R4 = ContinuousFunctionalNode(:R4, [model4], MonteCarlo(n2), discretization1)
R5 = ContinuousFunctionalNode(:R5, [model5], MonteCarlo(n2), discretization2)

model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.R4, df.R5, df.V, df.H), :G)
performance = df -> df.G
simulation = MonteCarlo(n2)
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
EnhancedBayesianNetworks.evaluate!(net)

# evidence1 = Dict(
#     :R5_d => Symbol([94.45444444444445, 116.67666666666666]),
#     :R4_d => Symbol([63.72984058368004, 72.22222222222223])
# )

# ϕ1 = infer(eebn, :E, evidence1)

# evidence2 = Dict(
#     :R5_d => Symbol([94.45444444444445, 116.67666666666666]),
#     :R4_d => Symbol([138.88888888888889, 161.11111111111111])
# )

# ϕ2 = infer(eebn, :E, evidence2)

# evidence3 = Dict(
#     :R5_d => Symbol([183.34333333333333, 205.56555555555556]),
#     :R4_d => Symbol([138.88888888888889, 161.11111111111111])
# )

# ϕ3 = infer(eebn, :E, evidence3)




# scenario = DiscreteRootNode(:S, Dict(:normal => 0.5, :abnormal => 0.5))

# Uᵣ = ContinuousRootNode(:Uᵣ, Normal())
# μ_gamma = 60
# cov_gamma = 0.2
# α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
# V = ContinuousRootNode(:V, Gamma(α, θ))

# μ_gumbel1 = 50
# cov_gumbel1 = 0.4
# μ_loc1, β1 = distribution_parameters(μ_gumbel1, cov_gumbel1 * μ_gumbel1, Gumbel)

# μ_gumbel2 = 100
# cov_gumbel2 = 0.4
# μ_loc2, β2 = distribution_parameters(μ_gumbel2, cov_gumbel2 * μ_gumbel2, Gumbel)
# states = Dict(
#     [:normal] => Gumbel(μ_loc1, β1),
#     [:abnormal] => Gumbel(μ_loc2, β2)
# )

# H = ContinuousChildNode(:H, [scenario], states)

# n = 10^6

# function plastic_moment_capacities(uᵣ)
#     ρ = 0.5477
#     μ = 150
#     cov = 0.2

#     λ, ζ = distribution_parameters(μ, μ * cov, LogNormal)

#     normal_μ = λ + ρ * ζ * uᵣ
#     normal_std = sqrt((1 - ρ^2) * ζ^2)
#     exp(rand(Normal(normal_μ, normal_std), 1)[1])
# end

# parents = [Uᵣ]
# model1 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r1)
# model2 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r2)
# model3 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r3)
# model4 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r4)
# model5 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r5)

# R1 = ContinuousFunctionalNode(:R1, parents, [model1], MonteCarlo(n))
# R2 = ContinuousFunctionalNode(:R2, parents, [model2], MonteCarlo(n))
# R3 = ContinuousFunctionalNode(:R3, parents, [model3], MonteCarlo(n))
# R4 = ContinuousFunctionalNode(:R4, parents, [model4], MonteCarlo(n))
# R5 = ContinuousFunctionalNode(:R5, parents, [model5], MonteCarlo(n))


# function frame_model(r1, r2, r3, r4, r5, v, h)
#     g1 = r1 + r2 + r4 + r5 - 5 * h
#     g2 = r2 + 2 * r3 + r4 - 5 * v
#     g3 = r1 + 2 * r3 + 2 * r4 + r5 - 5 * h - 5 * v
#     return minimum([g1, g2, g3])
# end

# model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.r4, df.r5, df.V, df.H), :G)
# performance = df -> df.G
# simulation = MonteCarlo(n)
# frame = DiscreteFunctionalNode(:E, [R1, R2, R3, R4, R5, V, H], [model], performance, simulation)


# nodes = [scenario, Uᵣ, V, H, R1, R2, R3, R4, R5, frame]
# ebn = EnhancedBayesianNetwork(nodes)

# eebn = EnhancedBayesianNetworks.evaluate(ebn)

# eebn.nodes[end].states
####

## Signle SRP
function plastic_moment_capacities(uᵣ)
    ρ = 0.5477
    μ = 150
    cov = 0.2

    λ, ζ = distribution_parameters(μ, μ * cov, LogNormal)

    normal_μ = λ + ρ * ζ * uᵣ
    normal_std = sqrt((1 - ρ^2) * ζ^2)
    exp(rand(Normal(normal_μ, normal_std), 1)[1])
end

u = RandomVariable(Normal(), :u)

μ_gamma = 60
cov_gamma = 0.2
α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
V = RandomVariable(Gamma(α, θ), :V)

μ_gumbel = 50
cov_gumbel = 0.4
μ_loc, β = distribution_parameters(μ_gumbel, cov_gumbel * μ_gumbel, Gumbel)
H = RandomVariable(Gumbel(μ_loc, β), :H)

model1 = Model(df -> plastic_moment_capacities.(df.u), :r1)
model2 = Model(df -> plastic_moment_capacities.(df.u), :r2)
model3 = Model(df -> plastic_moment_capacities.(df.u), :r3)
model4 = Model(df -> plastic_moment_capacities.(df.u), :r4)
model5 = Model(df -> plastic_moment_capacities.(df.u), :r5)

model_failure = Model(df -> frame_model.(df.r1, df.r2, df.r3, 150, 200, df.V, df.H), :G)
models = [model1, model2, model3, model4, model5, model_failure]

inputs = [u, V, H]

performance = df -> df.G

pf1 = UncertaintyQuantification.probability_of_failure(models, performance, inputs, MonteCarlo(10^6))
###



# model_failure = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.r4, df.r5, df.V, df.H), :G)
# models = [model1, model2, model3, model_failure]

# r4 = Parameter(50, :r4)
# r5 = Parameter(100, :r5)
# inputs = [u, V, H, r4, r5]

# performance = df -> df.G

# pf2 = UncertaintyQuantification.probability_of_failure(models, performance, inputs, MonteCarlo(10^6))




