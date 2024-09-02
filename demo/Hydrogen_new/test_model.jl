using UncertaintyQuantification
if Sys.isapple()
    include("/Users/andreaperin_macos/Documents/Code/Hydrogen_project/JuliaHyram/wrapper.jl")
elseif Sys.iswindows()
    include("D:/Code/Hydrogen_project/JuliaHyram/wrapper.jl")
else
    error("missing linux option")
end

t_amb = RandomVariable(Normal(288.15, 20), :t_amb)
p_amb = RandomVariable(Normal(101325, 5000), :p_amb)
t_h2 = RandomVariable(Normal(287.15, 20), :t_h2)
p_h2 = RandomVariable(Normal(13420000, 50000), :p_h2)

d_or = RandomVariable(Uniform(0.00356 - 0.003, 0.00356 + 0.003), :d_or)
Θ = RandomVariable(Uniform(50 / 180 * π, 130 / 180 * π), :Θ)
rel_humidity = RandomVariable(Uniform(0.85 - 0.1, 0.85 + 0.1), :humidity)

inputs = [t_amb, p_amb, t_h2, p_h2, d_or, Θ, rel_humidity]

n = 10

scenarios = rand(0:3, n)
discharge_coefficient = ones(n)
nozzle = repeat(["yuce"], n)
method = repeat(["bst"], n)

x = range(-30, 30, 101)
y = range(-30, 30, 101)
z = range(0, 4, 10)
a = collect(Iterators.product(x, y, z))
loc = vec(a)
locations = repeat([loc], n)

df = sample(inputs, n)

r_operators = repeat([10], n)
time = repeat([30], n)

df[:, :scenario] = scenarios
df[:, :disch_coeff] = discharge_coefficient
df[:, :nozzle] = nozzle
df[:, :loc] = locations
df[:, :method] = method
df[:, :r_operators] = r_operators
df[:, :time] = time

model = Model(df -> wrapper_model.(
        df.scenario,
        df.t_amb,
        df.p_amb,
        df.t_h2,
        df.p_h2,
        df.d_or,
        df.Θ,
        1,
        df.nozzle,
        df.humidity,
        [loc],
        df.method
    ), :res)

function peak_overpressure(df)
    peak_op = inputs -> maximum(inputs["overpressures"])
    peak_op.(df.res)
end

function peak_impulse(df)
    peak_imp = inputs -> maximum(inputs["impulses"])
    peak_imp.(df.res)
end

threshold_op = 4000
## 1 percent probability of lethal injury 

threshold_1stlevel_burn = 115 # [(kW/m2 ) 4/3 s]
threshold_2ndlevel_burn = 250 # [(kW/m2 ) 4/3 s]
threshold_3rdlevel_burn = 900 # [(kW/m2 ) 4/3 s]
# dose that would be needed to produce first-degree burns in 1% of the exposed population.

r_opeators = 10
time = 30

## Overpressure and Fire
model_mfr = Model(df -> mass_flow_rate_extractor(df), :mfr)
## Plume Specific
model_t_plume = Model(df -> T_extractor_plume(df), :T_plume)
model_massfracs_plume = Model(df -> mass_extractor_plume(df), :massf_plume)
model_molefracs_plume = Model(df -> mole_extractor_plume(df), :molef_plume)
## Fire Specific
th_dose_model = Model(df -> th_dose_extractor(df), :th_dose)
radius_0_1st_level_model = Model(df -> burn_radius_0(df, threshold_1stlevel_burn, [loc]), :burn_r_0_1st)
radius_0_2nd_level_model = Model(df -> burn_radius_0(df, threshold_2ndlevel_burn, [loc]), :burn_r_0_2nd)
radius_0_3rd_level_model = Model(df -> burn_radius_0(df, threshold_3rdlevel_burn, [loc]), :burn_r_0_3rd)

radius_C_1st_level_model = Model(df -> burn_radius_C(df, threshold_1stlevel_burn, [loc]), :burn_r_C_1st)
radius_C_2nd_level_model = Model(df -> burn_radius_C(df, threshold_2ndlevel_burn, [loc]), :burn_r_C_2nd)
radius_C_3th_level_model = Model(df -> burn_radius_C(df, threshold_3rdlevel_burn, [loc]), :burn_r_C_3th)

## Overpressure Specific
overpressures_model = Model(df -> overpressures_extractor(df), :op)
explosion_radius_0_model = Model(df -> explosion_radius_0(df, threshold_op, [loc]), :explosion_r_0)
explosion_radius_c_model = Model(df -> explosion_radius_C(df, threshold_op, [loc]), :explosion_r_C)
# evaluate!([model, model_op, model_imp, head_factor_model, explosion_radius_0_model, explosion_radius_c_model, fatalities_model], df)
evaluate!([model, model_mfr, model_t_plume, overpressures_model, th_dose_model, radius_0_1st_level_model, radius_0_2nd_level_model, radius_0_3rd_level_model, radius_C_1st_level_model, radius_C_2nd_level_model, radius_C_3th_level_model, explosion_radius_0_model, explosion_radius_c_model], df)