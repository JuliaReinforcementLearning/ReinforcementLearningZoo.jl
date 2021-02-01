export BehaviorCloingPolicy

# https://github.com/FluxML/Flux.jl/pull/1492
# TODO: use OneHotMatrix with logitcrossentropy instread
function logit_sparse_crossentropy(ŷ, y; agg=mean)
    agg(.-logsoftmax(ŷ)[CartesianIndex.(y, 1:length(y))])
end

"""
    BehaviorCloingPolicy(;kw...)

# Keyword Arguments

- `approximator`: calculate the logits of possible actions directly
- `loss_func=logit_sparse_crossentropy`
- `explorer=GreedyExplorer()` 

"""
Base.@kwdef struct BehaviorCloingPolicy{A} <: AbstractPolicy
    approximator::A
    loss_func::Any = logit_sparse_crossentropy
    explorer::Any = GreedyExplorer()
end

function (p::BehaviorCloingPolicy)(env::AbstractEnv)
    s = state(env)
    s_batch = Flux.unsqueeze(s, ndims(s) + 1)
    logits = p.approximator(s_batch) |> vec  # drop dimension
    p.explorer(logits)
end

function RLBase.update!(p::BehaviorCloingPolicy, batch::NamedTuple{(:state, :action)})
    s, a = batch.state, batch.action
    m, loss_func = p.approximator, p.loss_func
    gs = gradient(params(m)) do
        loss_func(m(s), a)
    end
    update!(m, gs)
end