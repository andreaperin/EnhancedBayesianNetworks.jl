function _verify_discretization(cpt::DataFrame, discretization::AbstractDiscretization)
    if _is_continuous_root(cpt) && isa(discretization, ApproximatedDiscretization)
        error("Root node must have ExactDiscretization as discretization structure, provided discretization is $discretization and node cpt is $cpt")
    elseif !_is_continuous_root(cpt) && isa(discretization, ExactDiscretization)
        error("Child node must have ExactDiscretization as discretization structure, provided discretization is $discretization and node cpt is $cpt")
    end
end