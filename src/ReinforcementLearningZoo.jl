module ReinforcementLearningZoo

const RLZoo = ReinforcementLearningZoo
export RLZoo

using ReinforcementLearningBase
using ReinforcementLearningCore

include("patch.jl")
include("algorithms/algorithms.jl")

end # module
