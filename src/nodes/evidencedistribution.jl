# struct EvidenceDistribution <: ContinuousUnivariateDistribution
#     parent_distribution::D where {D<:Distribution}
#     interval::Vector{<:Real}
#     normalization_factor::Real
#     function EvidenceDistribution(parent_distribution::D, interval::Vector{<:Real}) where {D<:Distribution}
#         normalization_factor = Distributions.cdf(parent_distribution, maximum(interval)) - Distributions.cdf(parent_distribution, minimum(interval))
#         new(parent_distribution, interval, normalization_factor)
#     end
# end

# function pdf(d::EvidenceDistribution, x::Real)
#     return insupport(d, x) ? Distributions.pdf(d.parent_distribution, x) / d.normalization_factor : zero(x)
# end

# function logpdf(d::EvidenceDistribution, x::Real)
#     return log(pdf(d, x))
# end

# function cdf(d::EvidenceDistribution, x::Real)
#     return clamp((Distributions.cdf(d.parent_distribution, x) - Distributions.cdf(d.parent_distribution, minimum(d.interval))) / d.normalization_factor, 0, 1)
# end

# function quantile(d::EvidenceDistribution, x::Real)
#     return ((Distributions.quantile(d.parent_distribution, Distributions.cdf(d.parent_distribution, minimum(d.interval)) + x * d.normalization_factor)))
# end


# function quantile(d::EvidenceDistribution, x::Real)
#     return d.quantile(x)
# end


# # function rand(rng::AbstractRNG, d::EvidenceDistribution)
# #     return quantile(d, rand(rng))
# # end

# insupport(d::EvidenceDistribution, x::Real) = minimum(d.interval) <= x <= maximum(d.interval)
