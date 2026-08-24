using EnhancedBayesianNetworks
using SpecialFunctions

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

# Carbonation-induced corrosion model

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

# Chloride-indeuced corrosion model

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
