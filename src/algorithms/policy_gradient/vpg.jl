using Distributions: Categorical, Normal
using Flux: normalise
using Random: shuffle

using ReinforcementLearningBase
using ReinforcementLearningCore

export VPGPolicy

"""
Vanilla Policy Gradient
"""
Base.@kwdef mutable struct VPGPolicy{
    A<:NeuralNetworkApproximator,
    B<:Union{NeuralNetworkApproximator,Nothing},
    R<:AbstractRNG,
} <: AbstractPolicy
    approximator::A
    baseline::B = nothing
    dist::Any = Categorical
    γ::Float32 = 0.99f0 # discount factor
    α_θ = 1.0f0 # step size of policy
    α_w = 1.0f0 # step size of baseline
    batch_size::Int = 1024
    rng::R = Random.GLOBAL_RNG
    loss::Float32 = 0.0f0
    baseline_loss::Float32 = 0.0f0
end

function (agent::Agent{<:VPGPolicy,<:AbstractTrajectory})(::Training{PreActStage}, env)
    action = agent.policy(env)
    push!(agent.trajectory; state = get_state(env), action = action)
    if ActionStyle(env) === FULL_ACTION_SET
        push!(agent.trajectory; legal_actions_mask = get_legal_actions_mask(env))
    end
    update!(agent.policy, agent.trajectory, get_actions(env))
    action
end

function (agent::Agent{<:VPGPolicy,<:AbstractTrajectory})(::Training{PostEpisodeStage}, env)
    action = agent.policy(env)
    push!(agent.trajectory; state = get_state(env), action = action)
    if ActionStyle(env) === FULL_ACTION_SET
        push!(agent.trajectory; legal_actions_mask = get_legal_actions_mask(env))
    end
    update!(agent.policy, agent.trajectory, get_actions(env))
    action
end

function (π::VPGPolicy)(env)
    to_dev(x) = send_to_device(device(π.approximator), x)

    logits = env |> get_state |> to_dev |> π.approximator
    π(logits, get_actions(env))
end

function (π::VPGPolicy)(env::MultiThreadEnv)
    error("not implemented")
    # TODO: can PG support multi env? PG only get updated at the end of an episode.
end

function (π::VPGPolicy)(logits, actions::DiscreteSpace)
    dist = logits |> softmax |> π.dist
    action = actions[rand(π.rng, dist)]
end

"""
See
* [Diagonal Gaussian Policies](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#stochastic-policies
* [Clipped Action Policy Gradient](https://arxiv.org/pdf/1802.07564.pdf)
"""
function (π::VPGPolicy)(logits, actions::ContinuousSpace)
    dist = π.dist(logits[1], exp(logits[2]))
    action = clamp(rand(π.rng, dist), actions.low, actions.high)
end

function (π::VPGPolicy)(logits, actions::MultiDiscreteSpace)
    error("not implemented")
    # TODO
end

function (π::VPGPolicy)(logits, actions::MultiContinuousSpace)
    error("not implemented")
    # TODO
end

function RLBase.update!(π::VPGPolicy, traj::ElasticCompactSARTSATrajectory)
    error("not supported")
end

function RLBase.update!(π::VPGPolicy, traj::ElasticCompactSARTSATrajectory, ::DiscreteSpace)
    (length(traj[:terminal]) > 0 && traj[:terminal][end]) || return

    model = π.approximator
    to_dev(x) = send_to_device(device(model), x)

    states = traj[:state]
    actions = traj[:action]
    gains = traj[:reward] |> x -> discount_rewards(x, π.γ)

    for idx in Iterators.partition(shuffle(1:length(traj[:terminal])), π.batch_size)
        S = states[:, idx] |> to_dev
        A = actions[idx]
        G = gains[idx] |> x -> Flux.unsqueeze(x, 1) |> to_dev
        # gains is a 1 colomn array, but the ouput of flux model is 1 row, n_batch columns array. so unsqueeze it.

        if typeof(π.baseline) <: NeuralNetworkApproximator
            δ = G - π.baseline(S)
            gs = gradient(Flux.params(π.baseline)) do
                # TODO: is the loss function correct?
                loss = mse(π.baseline(S), G) * π.α_w
                ignore() do
                    π.baseline_loss = loss
                end
                loss
            end
            update!(π.baseline, gs)
        end
        if typeof(π.baseline) <: Nothing
            # normalise should not be used with baseline. or the loss of the policy will be too small.
            δ = G |> x -> normalise(x; dims = 2)
        end

        gs = gradient(Flux.params(model)) do
            log_prob = S |> model |> logsoftmax
            log_probₐ = log_prob[CartesianIndex.(A, 1:length(A))]
            loss = -mean(log_probₐ .* δ) * π.α_θ
            ignore() do
                π.loss = loss
            end
            loss
        end
        update!(model, gs)
    end
    empty!(traj)
end

function RLBase.update!(
    π::VPGPolicy,
    traj::ElasticCompactSARTSATrajectory,
    ::ContinuousSpace,
)
    (length(traj[:terminal]) > 0 && traj[:terminal][end]) || return

    model = π.approximator
    to_dev(x) = send_to_device(device(model), x)

    states = traj[:state] |> to_dev
    actions = traj[:action]
    gains =
        traj[:reward] |>
        x -> discount_rewards(x, π.γ) |> x -> normalise(x; dims = 1) |> to_dev

    # TODO: fix bug
    gs = gradient(Flux.params(model)) do
        logits = states |> model
        @views μ, σ = logits[1, :], exp.(logits[2, :])
        dist = π.dist.(μ, σ)
        log_probₐ = logpdf.(dist, actions)
        loss = -mean(log_probₐ .* gains) * π.α
        ignore() do
            π.loss = loss
        end
        loss
    end

    update!(model, gs)
    empty!(traj)
end
