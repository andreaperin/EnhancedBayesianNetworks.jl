# # Climate-Dependent Corrosion of a Reinforced-Concrete Structure
#
# This example builds an *enhanced Bayesian Network* (eBN) for the deterioration
# of a reinforced-concrete element under two competing corrosion mechanisms —
# **carbonation-induced** and **chloride-induced** — whose drivers (temperature,
# relative humidity, atmospheric CO₂) evolve across the century under three
# shared socio-economic pathways (the SSP scenarios `ssp1`, `ssp2`, `ssp5`). The
# climate inputs are given as time slices from 2020 to 2100, and temperature is
# *imprecise* — an [`Interval`](@ref) per scenario — which makes the reduced
# network a credal one.
#
# The workflow is the usual eBN one: define the random inputs as root nodes, wire
# the physical models in as functional nodes, then [`reduce`](@ref) the network
# and run inference.

using EnhancedBayesianNetworks
using SpecialFunctions

# ## Global parameters
#
# Deterministic constants shared by the corrosion models — reference conditions,
# molar masses, cover thickness, and the intervention set-points for the
# dehumidifier and dechlorination systems.

RH_ref = Parameter(0.65, :RH_ref)
T_ref = Parameter(10 + 273.15, :T_ref)
M_CO₂ = Parameter(44, :M_CO₂)
M_CₐO = Parameter(56, :M_CₐO)
R = Parameter(8.314e-3, :R)
thickness = Parameter(0.3, :thickness)
t₀ = Parameter(2019, :t₀)
C_Ch_threshold = Parameter(0.9, :C_Ch_threshold)
threshold_carbonation = Parameter(3, :threshold_carbonation)            # [cm]
threshold_corrosion = Parameter(0.12, :threshold_corrosion)             # [m]

dehumidifier_threshold = Parameter(0.25, :dehumidifier_threshold)
dehumidifier_target = Parameter(0.2, :dehumidifier_target)
dechlorifier_threshold = Parameter(0.9, :dechlorifier_threshold)
dechlorifier_target = Parameter(0.7, :dechlorifier_threshold)

# ## Carbonation-induced corrosion model
#
# The functions below implement the carbonation front: how ambient humidity and
# CO₂ translate into a carbonation depth per time slice, and how those slices
# accumulate into a total corrosion depth. They are plain Julia functions here;
# later they are attached to the network as functional nodes.

function model_RH(RH_int, RH_ref)
    rand() > 0.5 ? r = 1 : r = -1
    l = RH_ref * RH_int
    RH = RH_ref - r * l
    if RH > 1
        RH = 1
    elseif RH < 0
        RH = 0
    end
    return RH
end

function model_ppm2kg(ppm)
    return ppm * 10^(-3)
end

function model_wc(ratio_wc_int)
    return ratio_wc_int * 0.1 + 0.5
end

function model_αₕ(ratio_wc)
    return 1 - ℯ^(-3.38 * ratio_wc)
end

function model_a(Cₑ, CₐO, αₕ, M_CO₂, M_CₐO)
    return 0.75 * Cₑ * CₐO * αₕ * M_CO₂ / M_CₐO
end

function model_fₜ(E, R, T_ref, T)
    return exp(E / R * (T_ref^-1 - T^-1))
end

function model_f_rh_carb(RH, RH_ref)
    fₑ = 2.5
    gₑ = 5
    if RH <= 0.25
        return 0
    else
        return ((1 - RH^fₑ) / (1 - RH_ref^fₑ))^gₑ
    end
end

function Δ_carbonation(k_site, t, fₜ, f_rh, D_i, a, n_d, CO₂, n_m)
    time_mapping = Dict(
        2020 => 1,
        2030 => 2,
        2040 => 3,
        2050 => 4,
        2060 => 5,
        2070 => 6,
        2080 => 7,
        2090 => 8,
        2100 => 9
    )
    return 100 * √(2 * k_site * D_i * (time_mapping[t])^(-n_d) / a * 10 * fₜ * f_rh * CO₂) * (1 / (time_mapping[t]))^(n_m)
end

function model_total_corrosion_depth(t, Δcarb_2020, Δcarb_2030, Δcarb_2040, Δcarb_2050, Δcarb_2060, Δcarb_2070, Δcarb_2080, Δcarb_2090, Δcarb_2100)
    if t == 2020
        return Δcarb_2020
    elseif t == 2030
        return Δcarb_2020 + Δcarb_2030
    elseif t == 2040
        return Δcarb_2020 + Δcarb_2030 + Δcarb_2040
    elseif t == 2050
        return Δcarb_2020 + Δcarb_2030 + Δcarb_2040 + Δcarb_2050
    elseif t == 2060
        return Δcarb_2020 + Δcarb_2030 + Δcarb_2040 + Δcarb_2050 + Δcarb_2060
    elseif t == 2070
        return Δcarb_2020 + Δcarb_2030 + Δcarb_2040 + Δcarb_2050 + Δcarb_2060 + Δcarb_2070
    elseif t == 2080
        return Δcarb_2020 + Δcarb_2030 + Δcarb_2040 + Δcarb_2050 + Δcarb_2060 + Δcarb_2070 + Δcarb_2080
    elseif t == 2090
        return Δcarb_2020 + Δcarb_2030 + Δcarb_2040 + Δcarb_2050 + Δcarb_2060 + Δcarb_2070 + Δcarb_2080 + Δcarb_2090
    elseif t == 2100
        return Δcarb_2020 + Δcarb_2030 + Δcarb_2040 + Δcarb_2050 + Δcarb_2060 + Δcarb_2070 + Δcarb_2080 + Δcarb_2090 + Δcarb_2100
    end
end

# ## Chloride-induced corrosion model
#
# The chloride mechanism instead diffuses chloride into the cover and tracks the
# depth at which the concentration crosses a corrosion-initiation threshold. The
# `erf`-based profile is where `SpecialFunctions` is needed.

function model_D_Ch0(D_Ch0_int)
    return (D_Ch0_int * 0.2 + 1) * 6.0e-12
end

function model_ke(ke_int)
    return ke_int * 0.155 + 0.924
end

function model_kc(kc_int)
    return kc_int * 0.7 + 2.4
end

function model_f_rh_ch(RH, RH_ref)
    return (1 + (1 - RH^4) / (1 - RH_ref^4))^(-1)
end

function model_ch_corrosion_FE(D_Ch0, fₜ, f_rh_ch, ke, kt, kc, C_Ch_B, t, thickness)
    x = range(0, thickness, 100)
    time_mapping = Dict(
        2020 => 2,
        2030 => 3,
        2040 => 4,
        2050 => 5,
        2060 => 6,
        2070 => 7,
        2080 => 8,
        2090 => 9,
        2100 => 10
    )
    D = D_Ch0 * fₜ * f_rh_ch * ke * kt * kc
    return map(p -> (C_Ch_B * (1 - erf(p / (2 * √(D * (10 * 365 * 24 * 3600 * time_mapping[t]))))), p), x)
end

function model_corrosion_penetration(C_Ch_spatial, C_Ch_threshold)
    failurepoints = filter(x -> x[1] > C_Ch_threshold, C_Ch_spatial)
    if isempty(failurepoints)
        return 0
    else
        return maximum(map(i -> i[2], failurepoints))
    end
end

function model_dehumidifier(RH, DeH, threshold_RH, target_RH)
    if RH > threshold_RH && DeH == true
        return target_RH
    else
        return RH
    end
end

function model_ch_corrosion_depth(
        t, ch_corrosion_depth_2020, ch_corrosion_depth_2030, ch_corrosion_depth_2040, ch_corrosion_depth_2050, ch_corrosion_depth_2060, ch_corrosion_depth_2070, ch_corrosion_depth_2080, ch_corrosion_depth_2090, ch_corrosion_depth_2100
    )
    if t == 2020
        return ch_corrosion_depth_2020
    elseif t == 2030
        return ch_corrosion_depth_2030
    elseif t == 2040
        return ch_corrosion_depth_2040
    elseif t == 2050
        return ch_corrosion_depth_2050
    elseif t == 2060
        return ch_corrosion_depth_2060
    elseif t == 2070
        return ch_corrosion_depth_2070
    elseif t == 2080
        return ch_corrosion_depth_2080
    elseif t == 2090
        return ch_corrosion_depth_2090
    elseif t == 2100
        return ch_corrosion_depth_2100
    end
end

function model_dechlorifier(Ch, DeCh, threshold_Ch, target_Ch)
    if Ch > threshold_Ch && DeCh == true
        return target_Ch
    else
        return Ch
    end
end


# ## Random inputs: the root nodes
#
# All uncertain inputs enter as root nodes. Many climate variables share the same
# shape — one distribution per SSP scenario — so a small helper builds a
# [`ContinuousNode`](@ref) conditional on the `:proj` (projection) parent and
# fills in its three scenario branches.

function continuous_on_proj(name::Symbol, values)   # values aligned to [:ssp1, :ssp2, :ssp5]
    n = ContinuousNode(name, [:proj])
    n[:proj => :ssp1] = values[1]
    n[:proj => :ssp2] = values[2]
    n[:proj => :ssp5] = values[3]
    return n
end

# ### Discrete roots: time, scenario, and interventions
#
# The discrete drivers are the `:t` time slice (nine decades, each state carrying
# its year as a [`Parameter`](@ref)), the `:proj` SSP scenario, and the two
# protective systems (`:deH` dehumidifier, `:deCh` dechlorination) — each almost
# surely working, with a small failure probability.

slices = [2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]

# time slice: uniform over the 9 years, each state carries the year as a Parameter on :t
time_slice = DiscreteNode(:t, [Symbol(y) => [Parameter(y, :t)] for y in slices])
for y in slices
    time_slice[:t => Symbol(y)] = 1 / length(slices)
end

# projections: uniform over the three SSP scenarios
projections_node = DiscreteNode(:proj)
projections_node[:proj => :ssp1] = 1 / 3
projections_node[:proj => :ssp2] = 1 / 3
projections_node[:proj => :ssp5] = 1 / 3

# dehumidifier / dechlorination working|broken, with boolean Parameters
deH_node = DiscreteNode(:deH, [:deh_working => [Parameter(true, :deH)], :deh_broken => [Parameter(false, :deH)]])
deH_node[:deH => :deh_working] = 1 - 10^(-4)
deH_node[:deH => :deh_broken] = 10^(-4)

deCh_node = DiscreteNode(:deCh, [:deCh_working => [Parameter(true, :deCh)], :deCh_broken => [Parameter(false, :deCh)]])
deCh_node[:deCh => :deCh_working] = 1 - 10^(-4)
deCh_node[:deCh => :deCh_broken] = 10^(-4)

# ### Temperature — imprecise, per scenario
#
# Temperature (in kelvin) is given as an [`Interval`](@ref) per scenario and time
# slice rather than a single distribution: the bounds come from an ensemble of
# climate projections. These imprecise nodes are what make the reduced network
# credal.

int_T_2020 = [(281.87229, 285.56085), (281.6916, 286.32021), (282.20046, 286.2733)]
int_T_2030 = [(282.0982, 287.07407), (282.2602, 286.84145), (281.5062, 286.7255)]
int_T_2040 = [(282.2752, 285.75486), (282.2368, 287.67743), (281.87123, 287.1479)]
int_T_2050 = [(282.19724, 286.30088), (282.1169, 286.56024), (282.63064, 287.7634)]
int_T_2060 = [(281.95066, 287.67291), (282.8809, 287.83233), (284.1444, 288.1458)]
int_T_2070 = [(282.46791, 286.31398), (282.1799, 287.89093), (283.5394, 289.1019)]
int_T_2080 = [(282.02297, 287.23992), (283.1048, 287.36554), (284.88885, 289.7104)]
int_T_2090 = [(281.30641, 288.77651), (282.549, 287.85182), (284.36582, 290.5262)]
int_T_2100 = [(281.7551, 287.43491), (283.0796, 288.43243), (285.14805, 292.2743)]

int_to_interval(v) = [Interval(t...) for t in v]
temperature_2020_node = continuous_on_proj(:T_2020, int_to_interval(int_T_2020))
temperature_2030_node = continuous_on_proj(:T_2030, int_to_interval(int_T_2030))
temperature_2040_node = continuous_on_proj(:T_2040, int_to_interval(int_T_2040))
temperature_2050_node = continuous_on_proj(:T_2050, int_to_interval(int_T_2050))
temperature_2060_node = continuous_on_proj(:T_2060, int_to_interval(int_T_2060))
temperature_2070_node = continuous_on_proj(:T_2070, int_to_interval(int_T_2070))
temperature_2080_node = continuous_on_proj(:T_2080, int_to_interval(int_T_2080))
temperature_2090_node = continuous_on_proj(:T_2090, int_to_interval(int_T_2090))
temperature_2100_node = continuous_on_proj(:T_2100, int_to_interval(int_T_2100))

# ### Atmospheric CO₂ — Normal, per scenario
#
# The CO₂ concentration (ppm) is Normal, one mean/COV series per SSP scenario and
# time slice, rising fastest under the high-emission `ssp5` pathway.

μ_CO₂1 = [412.5, 430.8, 440.2, 442.7, 441.7, 437.5, 431.6, 426.0, 420.9]
COV_CO₂1 = [0, 0.016, 0.016, 0.017, 0.01, 0.008, 0.009, 0.01, 0.011]
μ_CO₂2 = [412.5, 435.0, 460.8, 486.5, 508.9, 524.3, 531.1, 533.7, 538.4]
COV_CO₂2 = [0, 0.018, 0.016, 0.015, 0.016, 0.014, 0.012, 0.01, 0.008]
μ_CO₂3 = [412.5, 448.8, 489.4, 540.5, 603.5, 677.1, 758.2, 844.8, 935.9]
COV_CO₂3 = [0, 0.022, 0.018, 0.019, 0.015, 0.013, 0.013, 0.012, 0.01]

CO₂ = Dict()
for i in range(1, length(slices))
    CO₂[i] = [Normal(μ_CO₂1[i], μ_CO₂1[i] * COV_CO₂1[i]), Normal(μ_CO₂2[i], μ_CO₂2[i] * COV_CO₂2[i]), Normal(μ_CO₂3[i], μ_CO₂3[i] * COV_CO₂3[i])]
end

CO₂_2020_node = continuous_on_proj(:ppm_CO₂_2020, CO₂[1])
CO₂_2030_node = continuous_on_proj(:ppm_CO₂_2030, CO₂[2])
CO₂_2040_node = continuous_on_proj(:ppm_CO₂_2040, CO₂[3])
CO₂_2050_node = continuous_on_proj(:ppm_CO₂_2050, CO₂[4])
CO₂_2060_node = continuous_on_proj(:ppm_CO₂_2060, CO₂[5])
CO₂_2070_node = continuous_on_proj(:ppm_CO₂_2070, CO₂[6])
CO₂_2080_node = continuous_on_proj(:ppm_CO₂_2080, CO₂[7])
CO₂_2090_node = continuous_on_proj(:ppm_CO₂_2090, CO₂[8])
CO₂_2100_node = continuous_on_proj(:ppm_CO₂_2100, CO₂[9])

# ### Relative humidity — Normal, per scenario
#
# Interior relative humidity (as a fraction) is likewise Normal per scenario and
# time slice; the tabulated means are percentages, so they are divided by 100.

μ_H1 = [9.35, 9.52, 10.07, 11.66, 12.71, 12.47, 14.18, 14.16, 13.43]
COV_H1 = [0.606, 0.547, 0.564, 0.563, 0.522, 0.546, 0.532, 0.574, 0.668]
μ_H2 = [8.48, 9.18, 11.99, 13.43, 14.99, 16.03, 17.14, 17.86, 19.64]
COV_H2 = [0.758, 0.591, 0.552, 0.494, 0.493, 0.508, 0.46, 0.462, 0.44]
μ_H3 = [8.2, 11.08, 14.41, 16.62, 19.07, 23.01, 29.54, 32.8, 38.47]
COV_H3 = [0.61, 0.527, 0.491, 0.405, 0.381, 0.388, 0.385, 0.377, 0.373]

H = Dict()
for i in range(1, length(slices))
    H[i] = [Normal(μ_H1[i] / 100, μ_H1[i] / 100 * COV_H1[i]), Normal(μ_H2[i] / 100, μ_H2[i] / 100 * COV_H2[i]), Normal(μ_H3[i] / 100, μ_H3[i] / 100 * COV_H3[i])]
end

H_2020_node = continuous_on_proj(:RH_int_2020, H[1])
H_2030_node = continuous_on_proj(:RH_int_2030, H[2])
H_2040_node = continuous_on_proj(:RH_int_2040, H[3])
H_2050_node = continuous_on_proj(:RH_int_2050, H[4])
H_2060_node = continuous_on_proj(:RH_int_2060, H[5])
H_2070_node = continuous_on_proj(:RH_int_2070, H[6])
H_2080_node = continuous_on_proj(:RH_int_2080, H[7])
H_2090_node = continuous_on_proj(:RH_int_2090, H[8])
H_2100_node = continuous_on_proj(:RH_int_2100, H[9])

# ### Material and model constants
#
# The remaining continuous roots are scenario-independent material and model
# parameters: the CO₂ diffusion coefficient `:D_i`, the time exponents `:n_d` and
# `:n_m`, cement content, site and activation factors, and — further down — the
# chloride-model coefficients. Each is a single continuous root distribution.

μ_Dᵢ = 2.2e-4
σ_Dᵢ = 0.15e-4
CO₂_diff_node = ContinuousNode(:D_i, truncated(Normal(μ_Dᵢ, σ_Dᵢ), 0, Inf))

μ_nd = 0.24
σ_nd = 0.12 * μ_nd
n_d_node = ContinuousNode(:n_d, truncated(Normal(μ_nd, σ_nd), 0, Inf))

μ_nm = 0.12
σ_nm = 0.1 * μ_nm
n_m_node = ContinuousNode(:n_m, truncated(Normal(μ_nm, σ_nm), 0, Inf))

μ_Cₑ = 300
σ_Cₑ = 0.1 * μ_Cₑ
C_e_node = ContinuousNode(:Cₑ, truncated(Normal(μ_Cₑ, σ_Cₑ), 0, Inf))

μ_CₐO = 0.65
σ_CₐO = 0.1 * μ_CₐO
CₐO_node = ContinuousNode(:CₐO, Uniform(μ_CₐO - σ_CₐO, μ_CₐO + σ_CₐO))

μ_K_site = 1.15
σ_K_site = 0.1 * μ_K_site
K_site_node = ContinuousNode(:K_site, truncated(Normal(μ_K_site, σ_K_site), 0, Inf))

μ_E = 38.3
σ_E = 0.09 * μ_E
E_node = ContinuousNode(:E, Uniform(μ_E - σ_E, μ_E + σ_E))

μ_ratio_wc = 0.5
σ_ratio_wc = 0.05 * μ_ratio_wc
log_μ_ratio_wc, log_std_ratio_wc = distribution_parameters(μ_ratio_wc, σ_ratio_wc, LogNormal)
ratio_wc_node = ContinuousNode(:ratio_wc, truncated(LogNormal(log_μ_ratio_wc, log_std_ratio_wc), 0, 1))

# The chloride-specific coefficients: surface concentration, reference diffusion
# coefficient, and the environmental/ageing factors used by the chloride model.
μ_C_Ch_B = 1.15
σ_C_Ch_B = 0.675
C_Ch_B_node = ContinuousNode(:C_Ch_B1, truncated(Normal(μ_C_Ch_B, σ_C_Ch_B), 0, Inf))

log_μ_D_Ch0 = 0
log_std_D_Ch0 = 0.5
D_Ch0_node = ContinuousNode(:D_Ch0_int, truncated(LogNormal(log_μ_D_Ch0, log_std_D_Ch0), 0, 1))

ke_node = ContinuousNode(:ke_int, Gamma(2, 1))

μ_kt = 0.832
σ_kt = 0.024
kt_node = ContinuousNode(:kt, truncated(Normal(μ_kt, σ_kt), 0, Inf))

kc_node = ContinuousNode(:kc_int, Beta(2, 2))

# Collect every root node into a single vector, ready for the functional nodes,
# the network assembly, and `reduce`/`infer` (added in the next steps).

nodes = [
    time_slice, projections_node, deH_node, deCh_node,
    temperature_2020_node, temperature_2030_node, temperature_2040_node, temperature_2050_node,
    temperature_2060_node, temperature_2070_node, temperature_2080_node, temperature_2090_node, temperature_2100_node,
    CO₂_2020_node, CO₂_2030_node, CO₂_2040_node, CO₂_2050_node, CO₂_2060_node,
    CO₂_2070_node, CO₂_2080_node, CO₂_2090_node, CO₂_2100_node,
    H_2020_node, H_2030_node, H_2040_node, H_2050_node, H_2060_node,
    H_2070_node, H_2080_node, H_2090_node, H_2100_node,
    CO₂_diff_node, n_d_node, n_m_node, C_e_node, CₐO_node, K_site_node, E_node, ratio_wc_node,
    C_Ch_B_node, D_Ch0_node, kc_node, ke_node, kt_node,
]

# ## Physical models as functional nodes
#
# The corrosion physics enters the network as [`Model`](@ref)s (from
# `UncertaintyQuantification`) grouped into two [`DiscreteFunctionalNode`](@ref)s,
# one per mechanism. Each functional node runs a chain of models over the samples
# drawn from its parents and reduces the result to a *safe / failed* verdict
# through a performance function. Because temperature is imprecise, the inputs are
# imprecise too, so both nodes are evaluated with a [`DoubleLoop`](@ref) around a
# [`MonteCarlo`](@ref) simulation — the outer loop propagates the intervals, the
# inner one the aleatory distributions.
#
# `n` is kept deliberately small so the example builds quickly; a real study would
# use far more samples.

n = 10
sim_carbonation = DoubleLoop(MonteCarlo(n))
sim_corrosion = DoubleLoop(MonteCarlo(n))

# ### Carbonation branch
#
# The carbonation chain per time slice: map the raw humidity input to an ambient
# `RH`, convert CO₂ from ppm to kg/m³, apply the dehumidifier intervention, then
# the hydration degree `αₕ`, the CaO reaction term `a`, the temperature factor
# `fₜ`, and the humidity factor `f_rh_carb`. `Δ_carbonation` gives the depth added
# in each decade and `model_total_corrosion_depth` accumulates the slices up to the
# current year `t` into the total `x_c_tot`.

model_aux_rh_2020 = Model(df -> model_RH.(df.RH_int_2020, RH_ref.value), :RH1_2020)
model_aux_rh_2030 = Model(df -> model_RH.(df.RH_int_2030, RH_ref.value), :RH1_2030)
model_aux_rh_2040 = Model(df -> model_RH.(df.RH_int_2040, RH_ref.value), :RH1_2040)
model_aux_rh_2050 = Model(df -> model_RH.(df.RH_int_2050, RH_ref.value), :RH1_2050)
model_aux_rh_2060 = Model(df -> model_RH.(df.RH_int_2060, RH_ref.value), :RH1_2060)
model_aux_rh_2070 = Model(df -> model_RH.(df.RH_int_2070, RH_ref.value), :RH1_2070)
model_aux_rh_2080 = Model(df -> model_RH.(df.RH_int_2080, RH_ref.value), :RH1_2080)
model_aux_rh_2090 = Model(df -> model_RH.(df.RH_int_2090, RH_ref.value), :RH1_2090)
model_aux_rh_2100 = Model(df -> model_RH.(df.RH_int_2100, RH_ref.value), :RH1_2100)

model_aux_ppm_2020 = Model(df -> model_ppm2kg.(df.ppm_CO₂_2020), :Conc_CO₂_2020)
model_aux_ppm_2030 = Model(df -> model_ppm2kg.(df.ppm_CO₂_2030), :Conc_CO₂_2030)
model_aux_ppm_2040 = Model(df -> model_ppm2kg.(df.ppm_CO₂_2040), :Conc_CO₂_2040)
model_aux_ppm_2050 = Model(df -> model_ppm2kg.(df.ppm_CO₂_2050), :Conc_CO₂_2050)
model_aux_ppm_2060 = Model(df -> model_ppm2kg.(df.ppm_CO₂_2060), :Conc_CO₂_2060)
model_aux_ppm_2070 = Model(df -> model_ppm2kg.(df.ppm_CO₂_2070), :Conc_CO₂_2070)
model_aux_ppm_2080 = Model(df -> model_ppm2kg.(df.ppm_CO₂_2080), :Conc_CO₂_2080)
model_aux_ppm_2090 = Model(df -> model_ppm2kg.(df.ppm_CO₂_2090), :Conc_CO₂_2090)
model_aux_ppm_2100 = Model(df -> model_ppm2kg.(df.ppm_CO₂_2100), :Conc_CO₂_2100)

model_deh_2020 = Model(df -> model_dehumidifier.(df.RH1_2020, df.deH, dehumidifier_threshold.value, dehumidifier_target.value), :RH_2020)
model_deh_2030 = Model(df -> model_dehumidifier.(df.RH1_2030, df.deH, dehumidifier_threshold.value, dehumidifier_target.value), :RH_2030)
model_deh_2040 = Model(df -> model_dehumidifier.(df.RH1_2040, df.deH, dehumidifier_threshold.value, dehumidifier_target.value), :RH_2040)
model_deh_2050 = Model(df -> model_dehumidifier.(df.RH1_2050, df.deH, dehumidifier_threshold.value, dehumidifier_target.value), :RH_2050)
model_deh_2060 = Model(df -> model_dehumidifier.(df.RH1_2060, df.deH, dehumidifier_threshold.value, dehumidifier_target.value), :RH_2060)
model_deh_2070 = Model(df -> model_dehumidifier.(df.RH1_2070, df.deH, dehumidifier_threshold.value, dehumidifier_target.value), :RH_2070)
model_deh_2080 = Model(df -> model_dehumidifier.(df.RH1_2080, df.deH, dehumidifier_threshold.value, dehumidifier_target.value), :RH_2080)
model_deh_2090 = Model(df -> model_dehumidifier.(df.RH1_2090, df.deH, dehumidifier_threshold.value, dehumidifier_target.value), :RH_2090)
model_deh_2100 = Model(df -> model_dehumidifier.(df.RH1_2100, df.deH, dehumidifier_threshold.value, dehumidifier_target.value), :RH_2100)

model_alpha_h = Model(df -> model_αₕ.(df.ratio_wc), :αₕ)
model_a_ = Model(df -> model_a.(df.Cₑ, df.CₐO, df.αₕ, M_CO₂.value, M_CₐO.value), :a)

model_ft_2020 = Model(df -> model_fₜ.(df.E, R.value, T_ref.value, df.T_2020), :fₜ_2020)
model_ft_2030 = Model(df -> model_fₜ.(df.E, R.value, T_ref.value, df.T_2030), :fₜ_2030)
model_ft_2040 = Model(df -> model_fₜ.(df.E, R.value, T_ref.value, df.T_2040), :fₜ_2040)
model_ft_2050 = Model(df -> model_fₜ.(df.E, R.value, T_ref.value, df.T_2050), :fₜ_2050)
model_ft_2060 = Model(df -> model_fₜ.(df.E, R.value, T_ref.value, df.T_2060), :fₜ_2060)
model_ft_2070 = Model(df -> model_fₜ.(df.E, R.value, T_ref.value, df.T_2070), :fₜ_2070)
model_ft_2080 = Model(df -> model_fₜ.(df.E, R.value, T_ref.value, df.T_2080), :fₜ_2080)
model_ft_2090 = Model(df -> model_fₜ.(df.E, R.value, T_ref.value, df.T_2090), :fₜ_2090)
model_ft_2100 = Model(df -> model_fₜ.(df.E, R.value, T_ref.value, df.T_2100), :fₜ_2100)

model_f_rh_carb_2020 = Model(df -> model_f_rh_carb.(df.RH_2020, RH_ref.value), :f_rh_carb_2020)
model_f_rh_carb_2030 = Model(df -> model_f_rh_carb.(df.RH_2030, RH_ref.value), :f_rh_carb_2030)
model_f_rh_carb_2040 = Model(df -> model_f_rh_carb.(df.RH_2040, RH_ref.value), :f_rh_carb_2040)
model_f_rh_carb_2050 = Model(df -> model_f_rh_carb.(df.RH_2050, RH_ref.value), :f_rh_carb_2050)
model_f_rh_carb_2060 = Model(df -> model_f_rh_carb.(df.RH_2060, RH_ref.value), :f_rh_carb_2060)
model_f_rh_carb_2070 = Model(df -> model_f_rh_carb.(df.RH_2070, RH_ref.value), :f_rh_carb_2070)
model_f_rh_carb_2080 = Model(df -> model_f_rh_carb.(df.RH_2080, RH_ref.value), :f_rh_carb_2080)
model_f_rh_carb_2090 = Model(df -> model_f_rh_carb.(df.RH_2090, RH_ref.value), :f_rh_carb_2090)
model_f_rh_carb_2100 = Model(df -> model_f_rh_carb.(df.RH_2100, RH_ref.value), :f_rh_carb_2100)

model_Δ_carb_2020 = Model(df -> Δ_carbonation.(df.K_site, df.t, df.fₜ_2020, df.f_rh_carb_2020, df.D_i, df.a, df.n_d, df.Conc_CO₂_2020, df.n_m), :Δcarb_2020)
model_Δ_carb_2030 = Model(df -> Δ_carbonation.(df.K_site, df.t, df.fₜ_2030, df.f_rh_carb_2030, df.D_i, df.a, df.n_d, df.Conc_CO₂_2030, df.n_m), :Δcarb_2030)
model_Δ_carb_2040 = Model(df -> Δ_carbonation.(df.K_site, df.t, df.fₜ_2040, df.f_rh_carb_2040, df.D_i, df.a, df.n_d, df.Conc_CO₂_2040, df.n_m), :Δcarb_2040)
model_Δ_carb_2050 = Model(df -> Δ_carbonation.(df.K_site, df.t, df.fₜ_2050, df.f_rh_carb_2050, df.D_i, df.a, df.n_d, df.Conc_CO₂_2050, df.n_m), :Δcarb_2050)
model_Δ_carb_2060 = Model(df -> Δ_carbonation.(df.K_site, df.t, df.fₜ_2060, df.f_rh_carb_2060, df.D_i, df.a, df.n_d, df.Conc_CO₂_2060, df.n_m), :Δcarb_2060)
model_Δ_carb_2070 = Model(df -> Δ_carbonation.(df.K_site, df.t, df.fₜ_2070, df.f_rh_carb_2070, df.D_i, df.a, df.n_d, df.Conc_CO₂_2070, df.n_m), :Δcarb_2070)
model_Δ_carb_2080 = Model(df -> Δ_carbonation.(df.K_site, df.t, df.fₜ_2080, df.f_rh_carb_2080, df.D_i, df.a, df.n_d, df.Conc_CO₂_2080, df.n_m), :Δcarb_2080)
model_Δ_carb_2090 = Model(df -> Δ_carbonation.(df.K_site, df.t, df.fₜ_2090, df.f_rh_carb_2090, df.D_i, df.a, df.n_d, df.Conc_CO₂_2090, df.n_m), :Δcarb_2090)
model_Δ_carb_2100 = Model(df -> Δ_carbonation.(df.K_site, df.t, df.fₜ_2100, df.f_rh_carb_2100, df.D_i, df.a, df.n_d, df.Conc_CO₂_2100, df.n_m), :Δcarb_2100)

model_total_corrosion = Model(df -> model_total_corrosion_depth.(df.t, df.Δcarb_2020, df.Δcarb_2030, df.Δcarb_2040, df.Δcarb_2050, df.Δcarb_2060, df.Δcarb_2070, df.Δcarb_2080, df.Δcarb_2090, df.Δcarb_2100), :x_c_tot)

models_carbonation = [
    model_aux_rh_2020, model_aux_rh_2030, model_aux_rh_2040, model_aux_rh_2050, model_aux_rh_2060,
    model_aux_rh_2070, model_aux_rh_2080, model_aux_rh_2090, model_aux_rh_2100,
    model_aux_ppm_2020, model_aux_ppm_2030, model_aux_ppm_2040, model_aux_ppm_2050, model_aux_ppm_2060,
    model_aux_ppm_2070, model_aux_ppm_2080, model_aux_ppm_2090, model_aux_ppm_2100,
    model_deh_2020, model_deh_2030, model_deh_2040, model_deh_2050, model_deh_2060,
    model_deh_2070, model_deh_2080, model_deh_2090, model_deh_2100,
    model_alpha_h, model_a_,
    model_ft_2020, model_ft_2030, model_ft_2040, model_ft_2050, model_ft_2060,
    model_ft_2070, model_ft_2080, model_ft_2090, model_ft_2100,
    model_f_rh_carb_2020, model_f_rh_carb_2030, model_f_rh_carb_2040, model_f_rh_carb_2050, model_f_rh_carb_2060,
    model_f_rh_carb_2070, model_f_rh_carb_2080, model_f_rh_carb_2090, model_f_rh_carb_2100,
    model_Δ_carb_2020, model_Δ_carb_2030, model_Δ_carb_2040, model_Δ_carb_2050, model_Δ_carb_2060,
    model_Δ_carb_2070, model_Δ_carb_2080, model_Δ_carb_2090, model_Δ_carb_2100,
    model_total_corrosion,
]

# The performance is `threshold_carbonation − x_c_tot`: it turns positive when the
# accumulated carbonation depth stays below the allowable cover, negative on
# failure. The functional node therefore reports `carb_depth_safe` /
# `carb_depth_failed`.

carbonation_node = DiscreteFunctionalNode(:carb_depth, models_carbonation, df -> threshold_carbonation.value .- df.x_c_tot, sim_carbonation)

# ### Chloride branch
#
# The chloride chain reuses the humidity, CO₂ and dehumidifier auxiliaries, and
# adds the dechlorination intervention (`model_dech`), the diffusion-coefficient
# maps (`model_ke`, `model_kc`, `model_D_Ch0`), the humidity factor `f_rh_ch`, the
# `erf`-based finite-element chloride profile per slice (`model_ch_corrosion_FE`),
# the penetration depth where the concentration crosses the initiation threshold
# (`model_corrosion_penetration`), and finally the accumulated depth `x_ch_tot`.

model_dech = Model(df -> model_dechlorifier.(df.C_Ch_B1, df.deCh, dechlorifier_threshold.value, dechlorifier_target.value), :C_Ch_B)
model_aux_ke = Model(df -> model_ke.(df.ke_int), :ke)
model_aux_kc = Model(df -> model_kc.(df.kc_int), :kc)
model_aux_D_Ch0 = Model(df -> model_D_Ch0.(df.D_Ch0_int), :D_Ch0)

model_f_rh_ch_2020 = Model(df -> model_f_rh_ch.(df.RH_2020, RH_ref.value), :f_rh_ch_2020)
model_f_rh_ch_2030 = Model(df -> model_f_rh_ch.(df.RH_2030, RH_ref.value), :f_rh_ch_2030)
model_f_rh_ch_2040 = Model(df -> model_f_rh_ch.(df.RH_2040, RH_ref.value), :f_rh_ch_2040)
model_f_rh_ch_2050 = Model(df -> model_f_rh_ch.(df.RH_2050, RH_ref.value), :f_rh_ch_2050)
model_f_rh_ch_2060 = Model(df -> model_f_rh_ch.(df.RH_2060, RH_ref.value), :f_rh_ch_2060)
model_f_rh_ch_2070 = Model(df -> model_f_rh_ch.(df.RH_2070, RH_ref.value), :f_rh_ch_2070)
model_f_rh_ch_2080 = Model(df -> model_f_rh_ch.(df.RH_2080, RH_ref.value), :f_rh_ch_2080)
model_f_rh_ch_2090 = Model(df -> model_f_rh_ch.(df.RH_2090, RH_ref.value), :f_rh_ch_2090)
model_f_rh_ch_2100 = Model(df -> model_f_rh_ch.(df.RH_2100, RH_ref.value), :f_rh_ch_2100)

model_ch_corrosion_fe_2020 = Model(df -> model_ch_corrosion_FE.(df.D_Ch0, df.fₜ_2020, df.f_rh_ch_2020, df.ke, df.kt, df.kc, df.C_Ch_B, df.t, thickness.value), :fe_result_2020)
model_ch_corrosion_fe_2030 = Model(df -> model_ch_corrosion_FE.(df.D_Ch0, df.fₜ_2030, df.f_rh_ch_2030, df.ke, df.kt, df.kc, df.C_Ch_B, df.t, thickness.value), :fe_result_2030)
model_ch_corrosion_fe_2040 = Model(df -> model_ch_corrosion_FE.(df.D_Ch0, df.fₜ_2040, df.f_rh_ch_2040, df.ke, df.kt, df.kc, df.C_Ch_B, df.t, thickness.value), :fe_result_2040)
model_ch_corrosion_fe_2050 = Model(df -> model_ch_corrosion_FE.(df.D_Ch0, df.fₜ_2050, df.f_rh_ch_2050, df.ke, df.kt, df.kc, df.C_Ch_B, df.t, thickness.value), :fe_result_2050)
model_ch_corrosion_fe_2060 = Model(df -> model_ch_corrosion_FE.(df.D_Ch0, df.fₜ_2060, df.f_rh_ch_2060, df.ke, df.kt, df.kc, df.C_Ch_B, df.t, thickness.value), :fe_result_2060)
model_ch_corrosion_fe_2070 = Model(df -> model_ch_corrosion_FE.(df.D_Ch0, df.fₜ_2070, df.f_rh_ch_2070, df.ke, df.kt, df.kc, df.C_Ch_B, df.t, thickness.value), :fe_result_2070)
model_ch_corrosion_fe_2080 = Model(df -> model_ch_corrosion_FE.(df.D_Ch0, df.fₜ_2080, df.f_rh_ch_2080, df.ke, df.kt, df.kc, df.C_Ch_B, df.t, thickness.value), :fe_result_2080)
model_ch_corrosion_fe_2090 = Model(df -> model_ch_corrosion_FE.(df.D_Ch0, df.fₜ_2090, df.f_rh_ch_2090, df.ke, df.kt, df.kc, df.C_Ch_B, df.t, thickness.value), :fe_result_2090)
model_ch_corrosion_fe_2100 = Model(df -> model_ch_corrosion_FE.(df.D_Ch0, df.fₜ_2100, df.f_rh_ch_2100, df.ke, df.kt, df.kc, df.C_Ch_B, df.t, thickness.value), :fe_result_2100)

model_corrosion_ch_penetration_2020 = Model(df -> model_corrosion_penetration.(df.fe_result_2020, C_Ch_threshold.value), :ch_corrosion_depth_2020)
model_corrosion_ch_penetration_2030 = Model(df -> model_corrosion_penetration.(df.fe_result_2030, C_Ch_threshold.value), :ch_corrosion_depth_2030)
model_corrosion_ch_penetration_2040 = Model(df -> model_corrosion_penetration.(df.fe_result_2040, C_Ch_threshold.value), :ch_corrosion_depth_2040)
model_corrosion_ch_penetration_2050 = Model(df -> model_corrosion_penetration.(df.fe_result_2050, C_Ch_threshold.value), :ch_corrosion_depth_2050)
model_corrosion_ch_penetration_2060 = Model(df -> model_corrosion_penetration.(df.fe_result_2060, C_Ch_threshold.value), :ch_corrosion_depth_2060)
model_corrosion_ch_penetration_2070 = Model(df -> model_corrosion_penetration.(df.fe_result_2070, C_Ch_threshold.value), :ch_corrosion_depth_2070)
model_corrosion_ch_penetration_2080 = Model(df -> model_corrosion_penetration.(df.fe_result_2080, C_Ch_threshold.value), :ch_corrosion_depth_2080)
model_corrosion_ch_penetration_2090 = Model(df -> model_corrosion_penetration.(df.fe_result_2090, C_Ch_threshold.value), :ch_corrosion_depth_2090)
model_corrosion_ch_penetration_2100 = Model(df -> model_corrosion_penetration.(df.fe_result_2100, C_Ch_threshold.value), :ch_corrosion_depth_2100)

model_ch_corrosion = Model(df -> model_ch_corrosion_depth.(df.t, df.ch_corrosion_depth_2020, df.ch_corrosion_depth_2030, df.ch_corrosion_depth_2040, df.ch_corrosion_depth_2050, df.ch_corrosion_depth_2060, df.ch_corrosion_depth_2070, df.ch_corrosion_depth_2080, df.ch_corrosion_depth_2090, df.ch_corrosion_depth_2100), :x_ch_tot)

models_chloride = [
    model_aux_rh_2020, model_aux_rh_2030, model_aux_rh_2040, model_aux_rh_2050, model_aux_rh_2060,
    model_aux_rh_2070, model_aux_rh_2080, model_aux_rh_2090, model_aux_rh_2100,
    model_aux_ppm_2020, model_aux_ppm_2030, model_aux_ppm_2040, model_aux_ppm_2050, model_aux_ppm_2060,
    model_aux_ppm_2070, model_aux_ppm_2080, model_aux_ppm_2090, model_aux_ppm_2100,
    model_deh_2020, model_deh_2030, model_deh_2040, model_deh_2050, model_deh_2060,
    model_deh_2070, model_deh_2080, model_deh_2090, model_deh_2100,
    model_dech, model_aux_kc, model_aux_ke, model_aux_D_Ch0,
    model_ft_2020, model_ft_2030, model_ft_2040, model_ft_2050, model_ft_2060,
    model_ft_2070, model_ft_2080, model_ft_2090, model_ft_2100,
    model_f_rh_ch_2020, model_f_rh_ch_2030, model_f_rh_ch_2040, model_f_rh_ch_2050, model_f_rh_ch_2060,
    model_f_rh_ch_2070, model_f_rh_ch_2080, model_f_rh_ch_2090, model_f_rh_ch_2100,
    model_ch_corrosion_fe_2020, model_ch_corrosion_fe_2030, model_ch_corrosion_fe_2040, model_ch_corrosion_fe_2050, model_ch_corrosion_fe_2060,
    model_ch_corrosion_fe_2070, model_ch_corrosion_fe_2080, model_ch_corrosion_fe_2090, model_ch_corrosion_fe_2100,
    model_corrosion_ch_penetration_2020, model_corrosion_ch_penetration_2030, model_corrosion_ch_penetration_2040, model_corrosion_ch_penetration_2050, model_corrosion_ch_penetration_2060,
    model_corrosion_ch_penetration_2070, model_corrosion_ch_penetration_2080, model_corrosion_ch_penetration_2090, model_corrosion_ch_penetration_2100,
    model_ch_corrosion,
]

# As for carbonation, the performance is `threshold_corrosion − x_ch_tot`.

chloride_node = DiscreteFunctionalNode(:ch_corr_depth, models_chloride, df -> threshold_corrosion.value .- df.x_ch_tot, sim_corrosion)

# ## Assembling and wiring the network
#
# The two functional nodes join the roots to form the full node set. The edges
# encode the causal structure: the SSP `:proj` scenario drives every climate
# slice, and every driver (climate, material, model constants, time, and the
# relevant interventions) feeds the corrosion node of its mechanism.

network_nodes = [nodes; carbonation_node; chloride_node]
ebn = EnhancedBayesianNetwork(network_nodes)

# The scenario node conditions the temperature, humidity and CO₂ slices.

for slice_node in [
        temperature_2020_node, temperature_2030_node, temperature_2040_node, temperature_2050_node, temperature_2060_node,
        temperature_2070_node, temperature_2080_node, temperature_2090_node, temperature_2100_node,
        H_2020_node, H_2030_node, H_2040_node, H_2050_node, H_2060_node,
        H_2070_node, H_2080_node, H_2090_node, H_2100_node,
        CO₂_2020_node, CO₂_2030_node, CO₂_2040_node, CO₂_2050_node, CO₂_2060_node,
        CO₂_2070_node, CO₂_2080_node, CO₂_2090_node, CO₂_2100_node,
    ]
    add_child!(ebn, projections_node, slice_node)
end

# The carbonation node depends on all climate slices plus its material/model roots,
# the time slice, and the dehumidifier.

for parent in [
        temperature_2020_node, temperature_2030_node, temperature_2040_node, temperature_2050_node, temperature_2060_node,
        temperature_2070_node, temperature_2080_node, temperature_2090_node, temperature_2100_node,
        H_2020_node, H_2030_node, H_2040_node, H_2050_node, H_2060_node,
        H_2070_node, H_2080_node, H_2090_node, H_2100_node,
        CO₂_2020_node, CO₂_2030_node, CO₂_2040_node, CO₂_2050_node, CO₂_2060_node,
        CO₂_2070_node, CO₂_2080_node, CO₂_2090_node, CO₂_2100_node,
        E_node, CO₂_diff_node, n_d_node, n_m_node, C_e_node, CₐO_node, K_site_node, ratio_wc_node,
        time_slice, deH_node,
    ]
    add_child!(ebn, parent, carbonation_node)
end

# The chloride node depends on the same climate slices plus its own chloride
# coefficients, both interventions, and the time slice.

for parent in [
        temperature_2020_node, temperature_2030_node, temperature_2040_node, temperature_2050_node, temperature_2060_node,
        temperature_2070_node, temperature_2080_node, temperature_2090_node, temperature_2100_node,
        H_2020_node, H_2030_node, H_2040_node, H_2050_node, H_2060_node,
        H_2070_node, H_2080_node, H_2090_node, H_2100_node,
        CO₂_2020_node, CO₂_2030_node, CO₂_2040_node, CO₂_2050_node, CO₂_2060_node,
        CO₂_2070_node, CO₂_2080_node, CO₂_2090_node, CO₂_2100_node,
        E_node, deH_node, deCh_node, ke_node, kc_node, kt_node, D_Ch0_node, C_Ch_B_node,
        time_slice,
    ]
    add_child!(ebn, parent, chloride_node)
end

order!(ebn)

# ## Reducing to a credal network
#
# [`reduce`](@ref) orders the network, evaluates the functional nodes over their
# scenario grids (the expensive step — one imprecise simulation per scenario), and
# returns a discrete network. Because temperature is imprecise, the result is a
# [`CredalNetwork`](@ref); inference on it (via [`infer`](@ref)) then yields lower
# and upper failure probabilities.

reduced = reduce(ebn)
