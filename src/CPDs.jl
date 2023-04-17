using Distributions
using Discretizers
using UncertaintyQuantification


const global NodeName = Symbol
const global NodeNames = AbstractVector{NodeName}
abstract type CPD end

const global NodeName = Symbol
const global NodeNames = AbstractVector{NodeName}
const global ModelName = Symbol
const global AbstractDistribution = Union{Distribution,JointDistribution}

struct ModelWithName
    name::ModelName
    model::Vector{<:UQModel}
end

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

struct RootCPD <: CPD
    target::NodeName
    parents::NodeNames
    parental_ncategories::Vector{Int}
    distributions::Vector{<:AbstractDistribution}
    function RootCPD(target::NodeName, parents::NodeNames, parental_ncategories::Vector{Int}, distributions::Vector{<:AbstractDistribution})
        isempty(parents) ? new(target, parents, parental_ncategories, distributions) : throw(DomainError(target, "Is a RootNode with non empty parents argument"))
        isempty(parental_ncategories) ? new(target, parents, parental_ncategories, distributions) : throw(DomainError(target, "Is a RootNode with non empty parental_ncategories"))
        length(distributions) == 1 ? new(target, parents, parental_ncategories, distributions) : throw(DomainError(target, "Is a RootNode with more than 1 distribution"))
    end
end

RootCPD(target::NodeName, distributions::Vector{<:AbstractDistribution}) = RootCPD(target, NodeName[], Int[], distributions)
RootCPD(target::NodeName, distributions::D) where {D<:AbstractDistribution} = RootCPD(target, NodeName[], Int[], [distributions])

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

For Discrete and continuous nodes with known distributions and parents.
"""
struct StdCPD <: CPD
    target::NodeName
    parents::NodeNames
    parental_ncategories::Vector{Int}
    distributions::Vector{<:AbstractDistribution}

    function StdCPD(target::NodeName, parents::NodeNames, parental_ncategories::Vector{Int}, distribution::Vector{<:AbstractDistribution})
        prod(parental_ncategories) == length(distribution) ? new(target, parents, parental_ncategories, distribution) : throw(DomainError(target, "parental_ncategories-distributions length missmatch"))
    end
end

StdCPD(target::NodeName, parents::NodeNames, distributions::Vector{<:AbstractDistribution}) = StdCPD(target, parents, Int[], distributions)
StdCPD(target::NodeName, parents::NodeNames, distributions::D) where {D<:AbstractDistribution} = StdCPD(target, parents, Int[], [distributions])


name(cpd::StdCPD) = cpd.target
parents(cpd::StdCPD) = cpd.parents
nparams(cpd::StdCPD) = sum(d -> paramcount(params(d)), cpd.distributions)

"""
For Discrete and continuous nodes with AT LEAST one continuous parents and UNKNOWN Distribution. 
The Function or model return a SINGLE VALUE not a Distribution.
"""
struct FunctionalCPD <: CPD
    target::NodeName
    parents::NodeNames
    parental_ncategories::Vector{Int}
    distributions::Vector{<:ModelWithName}

    function FunctionalCPD(target::NodeName, parents::NodeNames, parental_ncategories::Vector{Int}, distributions::Vector{<:ModelWithName})
        prod(parental_ncategories) == length(distributions) ? new(target, parents, parental_ncategories, distributions) : throw(DomainError(target, "parental_ncategories-distributions length missmatch"))
    end
end


name(cpd::FunctionalCPD) = cpd.target
parents(cpd::FunctionalCPD) = cpd.parents
nparams(cpd::FunctionalCPD) = sum(d -> paramcount(params(d)), cpd.distributions)


function _get_type_of_cpd(cpd::Union{RootCPD,StdCPD})
    isa(cpd.distributions, Union{Vector{<:JointDistribution},Vector{<:ContinuousUnivariateDistribution}}) ? type = "continuous" : type = "discrete"
    return type
end