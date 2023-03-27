using Distributions
using Discretizers
using UncertaintyQuantification


const global NodeName = Symbol
const global NodeNames = AbstractVector{NodeName}
const global FunctionalModelCPD = Vector{M} where {M<:UQModel}
abstract type CPD end

"""
    An Object for mapping each distribution to a MapableTypes::Union{AbstractString, Symbol}
"""
const global MapableTypes = Union{AbstractString,Symbol}

struct MappedAliasTable <: Sampleable{Univariate,Discrete}
    alias::Distributions.AliasTable
    map::CategoricalDiscretizer
end

struct NamedCategorical{N<:MapableTypes} <: DiscreteUnivariateDistribution
    items::AbstractVector{N}
    probs::Vector{Float64}
    map::CategoricalDiscretizer{N,Int}
end

function NamedCategorical(item::AbstractVector{N}, probs::Vector{Float64}) where {N<:MapableTypes}
    map = CategoricalDiscretizer(item)
    sum(probs) == 1 ? probs = probs : probs = probs ./ sum(probs)
    NamedCategorical(item, probs, map)
end

Distributions.ncategories(s::MappedAliasTable) = Distributions.ncategories(s.alias)
Distributions.ncategories(d::NamedCategorical) = Distributions.ncategories(d.cat)

"""
A CPD for which the distribution never changes.
    target: name of the CPD's variable
    parents: list of parent variables.
    distributions: a Distributions.jl distribution
While a RootCPD can have parents, their assignments will not affect the distribution.
"""
struct RootCPD <: CPD
    target::NodeName
    parents::NodeNames
    parental_ncategories::Vector{Int}
    distributions::Vector{<:Distribution}
    function RootCPD(target::NodeName, parents::NodeNames, parental_ncategories::Vector{Int}, distributions::Vector{<:Distribution})
        isempty(parents) ? new(target, parents, parental_ncategories, distributions) : throw(DomainError(target, "Is a RootNode with non empty parents argument"))
        isempty(parental_ncategories) ? new(target, parents, parental_ncategories, distributions) : throw(DomainError(target, "Is a RootNode with non empty parental_ncategories"))
        length(distributions) == 1 ? new(target, parents, parental_ncategories, distributions) : throw(DomainError(target, "Is a RootNode with more than 1 distribution"))
    end
end

function RootCPD(target::NodeName, distributions::Vector{<:Distribution})
    RootCPD(target, NodeName[], Int[], distributions)
end

name(cpd::RootCPD) = cpd.target
parents(cpd::RootCPD) = cpd.parents

"""
A categorical CPD, P(x|parents(x)) where all parents are discrete integers 1:N 
and distributions ∀ combinations are known.
The ordering of `distributions` array follows the sequence: 
X,Y,Z
1,1,1
2,1,1
1,2,1
2,2,1
1,1,2
...

Discrete Parents ONLY
"""
struct StdCPD <: CPD
    target::NodeName
    parents::NodeNames
    parental_ncategories::Vector{Int}
    distributions::Vector{<:Distribution}

    function StdCPD(target::NodeName, parents::NodeNames, parental_ncategories::Vector{Int}, distribution::Vector{<:Distribution})
        length(parents) == length(parental_ncategories) ? new(target, parents, parental_ncategories, distribution) : throw(DomainError(target, "parents-parental_ncategories length missmatch"))
        prod(parental_ncategories) == length(distribution) ? new(target, parents, parental_ncategories, distribution) : throw(DomainError(target, "parental_ncategories-distributions length missmatch"))
    end
end

name(cpd::StdCPD) = cpd.target
parents(cpd::StdCPD) = cpd.parents
nparams(cpd::StdCPD) = sum(d -> paramcount(params(d)), cpd.distributions)


struct FunctionalCPD <: CPD
    target::NodeName
    parents::NodeNames
    parental_ncategories::Vector{Int}
    distributions::Vector{<:FunctionalModelCPD}

    function FunctionalCPD(target::NodeName, parents::NodeNames, parental_ncategories::Vector{Int}, distributions::Vector{<:FunctionalModelCPD})
        prod(parental_ncategories) == length(distributions) ? new(target, parents, parental_ncategories, distributions) : throw(DomainError(target, "parental_ncategories-distributions length missmatch"))
    end
end


name(cpd::FunctionalCPD) = cpd.target
parents(cpd::FunctionalCPD) = cpd.parents
nparams(cpd::FunctionalCPD) = sum(d -> paramcount(params(d)), cpd.distributions)


function _get_type_of_cpd(cpd::Union{RootCPD,StdCPD})
    isa(cpd.distributions, Vector{<:ContinuousUnivariateDistribution}) ? type = "continuous" : type = "discrete"
    return type
end