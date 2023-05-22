mutable struct Factor
    dimensions::Vector{Symbol}
    potential::Array{Float64} # Unnormalized probability
    # In most cases this will be a probability

    function Factor(dims::Vector{Symbol}, potential::Array{Float64})
        _ckeck_dims_unique(dims)

        (length(dims) != ndims(potential)) &&
            throw(DimensionMismatch("`potential` must have as many " *
                                    "dimensions as dims"))

        (:potential in dims) &&
            @warn("Having a dimension called `potential` will cause problems")

        return new(dims, potential)
    end
end