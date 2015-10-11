module IdealPoints
    import StatsFuns: logistic
    import Distributions: logpdf, Normal
    import Optim
    import RollCallDataIO
    import RollCallDataIO: ORDFile, SparseRollCall

    include("nd_model.jl")
end
