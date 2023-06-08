## Methods
Base.size(ϕ::Factor) = size(ϕ.potential)
Base.size(ϕ::Factor, dim::Symbol) = size(ϕ.potential, indexin(dim, ϕ))
Base.names(ϕ::Factor) = ϕ.dimensions
Base.in(dim::Symbol, ϕ::Factor) = dim in names(ϕ)
Base.indexin(dim::Symbol, ϕ::Factor) = findnext(isequal(dim), ϕ.dimensions, 1)
Base.indexin(dims::Vector{Symbol}, ϕ::Factor) = indexin(dims, names(ϕ))

Base.convert(::Type{Factor}, cpd::ConditionalProbabilityDistribution) = factorize_cpd(cpd)
Base.length(ϕ::Factor) = length(ϕ.potential)

Base.similar(ϕ::Factor) = Factor(ϕ.dimensions, similar(ϕ.potential), ϕ.states_mapping)


function Base.getindex(ϕ::Factor, e::Evidence)
    inds = _translate_index(ϕ, e)
    keep = inds .== Colon()
    new_dims = ϕ.dimensions[keep]
    @inbounds new_p = ϕ.potential[inds...]

    if ndims(new_p) == 0
        new_p = dropdims([new_p], dims=1)
    end

    states_mapping_dict = deepcopy(ϕ.states_mapping)
    delete!(states_mapping_dict, k for k in [filter(x -> x ∉ new_dims, ϕ.dimensions)])
    return Factor(new_dims, new_p, states_mapping_dict)
end