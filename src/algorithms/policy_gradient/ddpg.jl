export DDPGLearner

using Random
using Flux

mutable struct DDPGLearner{
    BA<:NeuralNetworkApproximator,
    BC<:NeuralNetworkApproximator,
    TA<:NeuralNetworkApproximator,
    TC<:NeuralNetworkApproximator,
    P,
    R<:AbstractRNG,
} <: AbstractPolicy

    behavior_actor::BA
    behavior_critic::BC
    target_actor::TA
    target_critic::TC
    γ::Float32
    ρ::Float32
    batch_size::Int
    start_steps::Int
    start_policy::P
    update_after::Int
    update_every::Int
    act_limit::Float64
    act_noise::Float64
    step::Int
    rng::R
end

"""
    DDPGLearner(;kwargs...)

# Keyword arguments

- `behavior_actor`,
- `behavior_critic`,
- `target_actor`,
- `target_critic`,
- `start_policy`,
- `γ = 0.99f0`,
- `ρ = 0.995f0`,
- `batch_size = 32`,
- `start_steps = 10000`,
- `update_after = 1000`,
- `update_every = 50`,
- `act_limit = 1.0`,
- `act_noise = 0.1`,
- `step = 0`,
- `rng = Random.GLOBAL_RNG`,
"""
function DDPGLearner(;
    behavior_actor,
    behavior_critic,
    target_actor,
    target_critic,
    start_policy,
    γ = 0.99f0,
    ρ = 0.995f0,
    batch_size = 32,
    start_steps = 10000,
    update_after = 1000,
    update_every = 50,
    act_limit = 1.0,
    act_noise = 0.1,
    step = 0,
    rng = Random.GLOBAL_RNG,
)
    copyto!(behavior_actor, target_actor)  # force sync
    copyto!(behavior_critic, target_critic)  # force sync
    DDPGLearner(
        behavior_actor,
        behavior_critic,
        target_actor,
        target_critic,
        γ,
        ρ,
        batch_size,
        start_steps,
        start_policy,
        update_after,
        update_every,
        act_limit,
        act_noise,
        step,
        rng,
    )
end

# TODO: handle Training/Testing mode
function (learner::DDPGLearner)(env)
    learner.step += 1

    if learner.step <= learner.start_steps
        learner.start_policy(env)
    else
        D = device(learner.behavior_actor)
        s = get_state(env)
        s = Flux.unsqueeze(s, ndims(s) + 1)
        action = learner.behavior_actor(send_to_device(D, s)) |> vec |> send_to_host
        clamp(action[] + randn(learner.rng) * learner.act_noise, -learner.act_limit, learner.act_limit)
    end
end

function RLBase.update!(learner::DDPGLearner, traj::CircularCompactSARTSATrajectory)
    length(traj[:terminal]) > learner.update_after || return
    learner.step % learner.update_every == 0 || return

    inds = rand(learner.rng, 1:(length(traj[:terminal])-1), learner.batch_size)
    s = select_last_dim(traj[:state], inds)
    a = select_last_dim(traj[:action], inds)
    r = select_last_dim(traj[:reward], inds)
    t = select_last_dim(traj[:terminal], inds)
    s′ = select_last_dim(traj[:next_state], inds)

    A = learner.behavior_actor
    C = learner.behavior_critic
    Aₜ = learner.target_actor
    Cₜ = learner.target_critic

    γ = learner.γ
    ρ = learner.ρ


    # !!! we have several assumptions here, need revisit when we have more complex environments
    # state is vector
    # action is scalar
    a′ = Aₜ(s′)
    qₜ = Cₜ(vcat(s′, a′)) |> vec
    y = r .+ γ .* (1 .- t) .* qₜ
    a = Flux.unsqueeze(a, 1)

    gs1 = gradient(Flux.params(C)) do
        q = C(vcat(s, a)) |> vec
        mean((y .- q) .^ 2)
    end

    update!(C, gs1)

    gs2 = gradient(Flux.params(A)) do
        -mean(C(vcat(s, A(s))))
    end

    update!(A, gs2)

    # polyak averaging
    for (dest, src) in zip(Flux.params([Aₜ, Cₜ]), Flux.params([A, C]))
        dest .= ρ .* dest .+ (1 - ρ) .* src
    end
end
