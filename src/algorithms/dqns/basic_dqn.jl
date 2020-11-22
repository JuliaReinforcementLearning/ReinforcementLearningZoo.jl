export BasicDQNLearner

using Random
using Flux

"""
    BasicDQNLearner(;kwargs...)

See paper: [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

This is the very basic implementation of DQN. Compared to the traditional Q learning, the only difference is that,
in the updating step it uses a batch of transitions sampled from an experience buffer instead of current transition.
And the `approximator` is usually a [`NeuralNetworkApproximator`](@ref).
You can start from this implementation to understand how everything is organized and how to write your own customized algorithm.
# Keywords
- `approximator`::[`AbstractApproximator`](@ref): used to get Q-values of a state.
- `loss_func`: the loss function to use. TODO: provide a default [`huber_loss`](@ref)?
- `γ::Float32=0.99f0`: discount rate.
- `batch_size::Int=32`
- `min_replay_history::Int=32`: number of transitions that should be experienced before updating the `approximator`.
- `rng=Random.GLOBAL_RNG`
"""
mutable struct BasicDQNLearner{Q,F,R} <: AbstractLearner
    approximator::Q
    loss_func::F
    γ::Float32
    batch_size::Int
    min_replay_history::Int
    rng::R
    loss::Float32
end

Flux.functor(x::BasicDQNLearner) = (Q = x.approximator,), y -> begin
    x = @set x.approximator = y.Q
    x
end

(learner::BasicDQNLearner)(env) =
    env |>
    get_state |>
    x ->
        send_to_device(device(learner.approximator), x) |>
        learner.approximator |>
        send_to_host

function BasicDQNLearner(;
    approximator::Q,
    loss_func::F,
    γ = 0.99f0,
    batch_size = 32,
    min_replay_history = 32,
    rng = Random.GLOBAL_RNG,
) where {Q,F}
    BasicDQNLearner{Q,F,typeof(rng)}(
        approximator,
        loss_func,
        γ,
        batch_size,
        min_replay_history,
        rng,
        0.0,
    )
end

function RLBase.update!(learner::BasicDQNLearner, T::AbstractTrajectory)
    length(T[:terminal]) < learner.min_replay_history && return

    inds = rand(learner.rng, 1:length(T[:terminal]), learner.batch_size)

    batch = (
        state = consecutive_view(T[:state], inds),
        action = consecutive_view(T[:action], inds),
        reward = consecutive_view(T[:reward], inds),
        terminal = consecutive_view(T[:terminal], inds),
        next_state = consecutive_view(T[:next_state], inds),
    )

    update!(learner, batch)
end

function RLBase.update!(learner::BasicDQNLearner, batch::NamedTuple)

    Q = learner.approximator
    D = device(Q)
    γ = learner.γ
    loss_func = learner.loss_func

    batch_size = nframes(batch.terminal)

    s = send_to_device(D, batch.state)
    a = batch.action
    r = send_to_device(D, batch.reward)
    t = send_to_device(D, batch.terminal)
    s′ = send_to_device(D, batch.next_state)

    a = CartesianIndex.(a, 1:batch_size)

    gs = gradient(params(Q)) do
        q = Q(s)[a]
        q′ = vec(maximum(Q(s′); dims = 1))
        G = r .+ γ .* (1 .- t) .* q′
        loss = loss_func(G, q)
        ignore() do
            learner.loss = loss
        end
        loss
    end

    update!(Q, gs)
end
