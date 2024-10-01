mutable struct Factor
    dimensions::Vector{Symbol}
    potential::Array
    states_mapping::Dict{Symbol,Dict{Symbol,Int}}

    function Factor(dims::Vector{Symbol}, potential::Array, states_mapping::Dict{Symbol,Dict{Symbol,Int}})
        _ckeck_dims_unique(dims)
        (length(dims) != ndims(potential)) && error("potential must have as many dimensions as length of dimensions")
        (:potential in dims) && error("Having a dimension called potential will cause problems")
        Set(dims) != Set(keys(states_mapping)) && error("states mapping keys have to be coherent with defined dimensions")
        return new(dims, potential, states_mapping)
    end
end

function _check_dims_valid(dims::Vector{Symbol}, ϕ::Factor)
    isempty(dims) && return
    dim = first(dims)
    (dim in ϕ) || error("Dimension is not in the factor")
    return _check_dims_valid(dims[2:end], ϕ)
end
# dims are unique
_ckeck_dims_unique(dims::Vector{Symbol}) = allunique(dims) || error("Dimensions must be unique")

# ## Create a factor for a node, given some evidence.
function Factor(bn::BayesianNetwork, node::Symbol, e::Evidence=Evidence())
    cpd = get_cpd(bn, node)
    ϕ = factorize_cpd(cpd)
    return ϕ[e]
end

## Convert cpd to factors
function factorize_cpd(cpd::ConditionalProbabilityDistribution)
    dims = vcat(cpd.target, cpd.parents)
    lengths = tuple(length(cpd.states), cpd.parental_ncategories...)
    p = Array{Float64}(undef, lengths)
    ## Create the states
    indices = sort(vcat(Iterators.product(map(x -> 1:x, cpd.parental_ncategories)...)...), by=last) |> collect
    f_m = (i, v) -> [k for (k, h) in cpd.parents_states_mapping_dict[cpd.parents[i]] if h == v]
    indices_new = map(ind -> vcat([f_m(i, v) for (i, v) in enumerate(ind)]...), indices)
    p[:] = vcat([collect(values(cpd.distribution[i])) for i in indices_new]...)

    states_mapping = deepcopy(cpd.parents_states_mapping_dict)
    states_mapping[cpd.target] = Dict(s => i for (i, s) in enumerate(cpd.states))
    return Factor(dims, p, states_mapping)
end

Base.convert(::Type{Factor}, cpd::ConditionalProbabilityDistribution) = factorize_cpd(cpd)

@inline function _translate_index(ϕ::Factor, e::Evidence)
    inds = Array{Any}(undef, length(ϕ.dimensions))
    inds[:] .= Colon()

    for (i, dim) in enumerate(ϕ.dimensions)
        if haskey(e, dim)
            ind = ϕ.states_mapping[dim][e[dim]]
            inds[i] = ind
        end
    end
    return inds
end


