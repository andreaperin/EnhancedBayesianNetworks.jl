function transfer_continuous(ebn::EnhancedBayesianNetwork)
    EnhancedBayesianNetwork(_transfer_continuous!(deepcopy(ebn.nodes)))
end
