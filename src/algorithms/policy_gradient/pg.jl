using Random
using Flux

using ReinforcementLearningBase
using ReinforcementLearningCore
using Distributions

export PGPolicy
Base.@kwdef mutable struct PGPolicy{A<:NeuralNetworkApproximator,R<:AbstractRNG} <:
                           AbstractPolicy
    approximator::A
    dist_fn::Any = Normal # distribution function
    γ::Float32 = 0.99f0 # discount factor
    rng::R = Random.GLOBAL_RNG
    loss::Float32 = 0f0
end

# if the action space is discrete, then softmax should be the last layer of the model.

function (π::PGPolicy)(env)
    to_dev = x -> send_to_device(device(π.approximator), x)

    logits = env |> get_state |> to_dev |> π.approximator |> send_to_host # if the action space is discrete, then the approximator should softmax the output values.
    dist = logits |> π.dist_fn

    actions = get_actions(env)
    if typeof(actions) <: DiscreteSpace
        action = actions[rand(dist)]
    elseif typeof(actions) <: ContinuousSpace
        error("Not implemented")
        action = clamp(rand(dist), actions.low, actions.high)
    else
        error("Not implemented")
    end
    action
end

function RLBase.update!(π::PGPolicy, traj::ElasticCompactSARTSATrajectory)
    (length(traj[:terminal]) > 0 && traj[:terminal][end]) || return

    Q = π.approximator
    D = device(Q)
    to_dev = x -> send_to_device(D, x)

    states = traj[:state] |> to_dev
    actions = traj[:action] |> Array
    rewards = traj[:reward] |> Array

    gains = discount_rewards(rewards, π.γ) |> to_dev |> x -> Flux.normalise(x; dims = 1)

    gs = gradient(Flux.params(Q)) do
        logits = states |> Q |> Array
        log_prob = logits[CartesianIndex.(actions, 1:length(actions))] .|> log
        entropy_loss = -mean(log_prob .* gains)
        ignore() do
            π.loss = entropy_loss
        end
        entropy_loss
    end

    update!(Q, gs)
    empty!(traj)
end
