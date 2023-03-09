using Distributions
using Discretizers
using UncertaintyQuantification

"""
    Definition of the NodeName constant
"""
const global NodeName = Symbol
const global NodeNames = AbstractVector{NodeName}

# abstract type SystemReliabilityProblem end
"""
    Definition of the CPD AbstractType
"""
abstract type CPD end

"""
    Definition of the SRP Struct
"""
struct CorrelationCopula
    nodes::Any
    copula::Union{GaussianCopula,Nothing}
    name::NodeName
end

function CorrelationCopula()
    nodes = Vector{NodeName}()
    copula = nothing
    name = NodeName()
    CorrelationCopula(nodes, copula, name)
end

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
    # cat::Categorical
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
    distributions::Union{NamedCategorical,ContinuousUnivariateDistribution}
    function RootCPD(target::NodeName, parents::NodeNames, distributions::Union{NamedCategorical,ContinuousUnivariateDistribution})
        isempty(parents) ? new(target, parents, distributions) : throw(DomainError(target, "Is a RootNode with non empty parents argument"))
    end
end

function RootCPD(target::NodeName, distributions::Union{NamedCategorical,ContinuousUnivariateDistribution})
    RootCPD(target, NodeName[], distributions)
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
"""
struct CategoricalCPD <: CPD
    target::NodeName
    parents::NodeNames
    parental_ncategories::Vector{Int}
    distributions::Vector{Distribution}
    function CategoricalCPD(target::NodeName, parents::NodeNames, parental_ncategories::Vector{Int}, distributions::Vector{D}) where {D<:Distribution}
        ##TODO this check should be done in node
        # length(parental_ncategories) == length(parents) ? new(target, parents, parental_ncategories, distributions) : throw(DomainError(target, "length of parental_ncategories is different from the number of defined parents"))
        prod(parental_ncategories) == length(distributions) ? new(target, parents, parental_ncategories, distributions) : throw(DomainError(target, "number of parental_ncategories is different from the number of  defined functions"))
    end

    ##TODO ALL of this should be done in node
    ## Creating prob_dict for CategoricalCPD
    function CategoricalCPD(target::NodeName, parents::NodeNames, parental_ncategories::Vector{Int}, distributions::Vector{D}) where {D<:Distribution}
        f = x -> collect(1:1:x)
        fn = x -> Dict(x.items .=> x.probs / sum(x.probs))
        fd = x -> Dict(:all_states .=> x)
        combinations = sort(vec(collect(Iterators.product(f.(parental_ncategories)...))))
        evidences_vector = map_combination2evidence.(combinations, repeat([parents], length(combinations)))
        if isa(distributions, Vector{NamedCategorical{Symbol}})
            prob_dict = CPDProbabilityDictionary.(tuple.(evidences_vector, fn.(distributions)))
        else
            prob_dict = CPDProbabilityDictionary.(tuple.(evidences_vector, fd.(distributions)))
        end
        new(target, parents, parental_ncategories, distributions, prob_dict)
    end
end

function map_combination2evidence(combination::Tuple, nodes::NodeNames)
    evidence_dict = Dict()
    if length(combination) != length(nodes)
        throw(DomainError([combination, nodes], "Assigned parents are not equals to the number of parental_ncategories"))
    else
        for i in range(1, length(combination))
            evidence_dict[nodes[i]] = combination[i]
        end
    end
    return evidence_dict
end

name(cpd::CategoricalCPD) = cpd.target
parents(cpd::CategoricalCPD) = cpd.parents


struct new_functional_cpd <: CPD
    target::NodeName
    parents::NodeNames
    parental_ncategorie::Vector{Int}
    distributions::Vector
    correlation::CorrelationCopula
end



# """
# A Parent Functional CPD to be used when there at least a continuos parents and/or distribution are not known.
# """

# struct FunctionalCPD <: CPD
#     target::NodeName
#     parents::NodeNames
#     parental_ncategories::Vector{Int}
#     prob_dict::Vector{CPDProbabilityDictionaryFunctional}
#     ## Check:
#     #    - parental_ncategories - prob_dict coherence
#     function FunctionalCPD(target::NodeName, parents::NodeNames, parental_ncategories::Vector{Int64}, prob_dict::Vector{CPDProbabilityDictionaryFunctional})
#         prod(parental_ncategories) == length(prob_dict) ? new(target, parents, parental_ncategories, prob_dict) : throw(DomainError(target, "number of parental_ncategories is different from the number of  defined functions"))
#     end
# end

# name(cpd::FunctionalCPD) = cpd.target
# parents(cpd::FunctionalCPD) = cpd.parents
