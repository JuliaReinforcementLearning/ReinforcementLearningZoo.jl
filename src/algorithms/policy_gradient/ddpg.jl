export DDPGPolicy

mutable struct DDPGPolicy{B, T, P, R} <: AbstractPolicy
    behavior_approximator::B
    target_approximator::T
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

function DDPGPolicy(;
        behavior_approximator,
        target_approximator,
        start_policy,
        γ=0.99f0,
        ρ=0.995f0,
        batch_size=32,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        act_limit=1.0,
        act_noise=0.1,
        step=0,
        seed=nothing
)
    rng = MersenneTwister(seed)
    copyto!(behavior_approximator, target_approximator)  # force sync
    DDPGPolicy(
        behavior_approximator,
        target_approximator,
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
        rng
    )
end

actor(app::NeuralNetworkApproximator{HYBRID_APPROXIMATOR, <:ActorCritic}) = app.model.actor
critic(app::NeuralNetworkApproximator{HYBRID_APPROXIMATOR, <:ActorCritic}) = app.model.critic

function (p::DDPGPolicy)(obs)
    p.step += 1
    
    if p.step <= p.start_steps
        p.start_policy(obs)
    else
        D = device(p.behavior_approximator)
        action = actor(p.behavior_approximator)(send_to_device(D, get_state(obs)))
        clamp(send_to_host(action)[1] + randn(p.rng) * p.act_noise, -p.act_limit, p.act_limit)
    end
end

function RLBase.update!(p::DDPGPolicy, t::CircularCompactSARTSATrajectory)
    length(t) > p.update_after || return
    p.step % p.update_every == 0 || return
    
    inds = rand(p.rng, 1:length(t), p.batch_size)
    SARTS = (:state, :action, :reward, :terminal, :next_state)
    s, a, r, t, s′= map(x -> select_last_dim(get_trace(t, x), inds), SARTS)
    a = Flux.unsqueeze(a, 1)

    A = actor(p.behavior_approximator)
    C = critic(p.behavior_approximator)
    Aₜ = actor(p.target_approximator)
    Cₜ = critic(p.target_approximator)

    γ = p.γ
    ρ = p.ρ

    
    a′ = Aₜ(s′)
    qₜ = Cₜ(vcat(s′, a′)) |> vec
    y = r .+ γ.*(1 .- t) .* qₜ


    gs1 = gradient(Flux.params(C)) do
        q = C(vcat(s, a)) |> vec
        mean((y .- q) .^ 2)
    end

    
    Flux.Optimise.update!(p.behavior_approximator.optimizer, Flux.params(C), gs1)
 
    gs2 = gradient(Flux.params(A)) do
        -mean(C(vcat(s, A(s))))
    end
    
    Flux.Optimise.update!(p.behavior_approximator.optimizer, Flux.params(A), gs2)
    
    # polyak averaging
    for (dest, src) in zip(Flux.params(p.target_approximator), Flux.params(p.behavior_approximator))
        dest .= ρ .* dest  .+ (1-ρ) .* src
    end
end