using ReinforcementLearningCore

(app::NeuralNetworkApproximator)(args...; kwargs...) = app.model(args...; kwargs...)

using AbstractTrees
using TensorBoardLogger: TBLogger

RLCore.is_expand(::TBLogger) = false
