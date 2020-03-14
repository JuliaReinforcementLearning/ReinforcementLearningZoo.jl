include("ppo_trajectory.jl")

export PPOLearner

struct PPOLearner{A, R} <: AbstractLearner
    approximator::A
    γ::Float32
    λ::Float32
    clip_range::Float32
    n_minibatch::Int
    minibatch_size::Int
    actor_loss_weight::Float32
    critic_loss_weight::Float32
    entropy_loss_weight::Float32
    rng::R
end

function PPOLearner(;
    approximator,
    γ=0.99f0,
    λ=0.95f0,
    clip_range=0.2f0,
    n_minibatch=4,
    minibatch_size=32,
    actor_loss_weight=1.0f0,
    critic_loss_weight=0.5f0,
    entropy_loss_weight=0.01f0,
    seed=nothing
)
    rng = MersenneTwister(seed)
    PPOLearner(
        approximator,
        γ,
        λ,
        clip_range,
        n_minibatch,
        minibatch_size,
        actor_loss_weight,
        critic_loss_weight,
        entropy_loss_weight,
        rng
    )
end

(learner::PPOLearner)(obs::BatchObs) =
    learner.approximator(
        send_to_device(device(learner.approximator), get_state(obs)),
        Val(:Q),
    ) |> send_to_host

function RLBase.update!(learner::PPOLearner, experience)
    states, actions,action_log_probs, rewards,terminals,states_plus = experience
    rng = learner.rng
    AC = learner.approximator
    γ = learner.γ
    λ = learner.λ
    mini_batch_size = learner.minibatch_size
    clip_range = learner.clip_range
    w₁ = learner.actor_loss_weight
    w₂ = learner.critic_loss_weight
    w₃ = learner.entropy_loss_weight
    D = device(AC)
    
    n_envs, n_rollout = size(terminals)
    states_flatten = flatten_batch(states)
    states_plus_flatten = flatten_batch(states_plus)
    states_plus_values = reshape(send_to_host(AC(send_to_device(D, states_plus_flatten), Val(:V))), n_envs,:) 
    advantages = generalized_advantage_estimation(rewards, states_plus_values, γ, λ;dims=2)
    returns = advantages .+ select_last_dim(states_plus_values, 1:n_rollout)

    for _ in 1:learner.n_minibatch
        inds = rand(rng, 1:nframes(states_flatten), mini_batch_size)
        s = send_to_device(D, select_last_dim(states_flatten, inds) |> copy)  # !!!
        a = vec(actions)[inds]
        r = send_to_device(D, vec(returns)[inds])
        log_p = send_to_device(D, vec(action_log_probs)[inds])
        adv = send_to_device(D, vec(advantages)[inds])

        gs = gradient(Flux.params(AC)) do
            v′ = AC(s, Val(:V)) |> vec
            p′ = AC(s, Val(:Q))
            log_p′ = log.(p′)
            log_p′ₐ = log_p′[CartesianIndex.(a, 1:length(a))]

            ratio = exp.(log_p′ₐ .- log_p)
            #  Zygote.dropgrad(println(ratio))
            surr1 = ratio .* adv
            surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

            actor_loss = - mean(min.(surr1, surr2))
            critic_loss = mean((r .- v′).^2)
            entropy_loss = sum(p′ .* log_p′) * 1 // size(p′, 2)
            loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss
            loss
        end

        update!(AC, gs)
    end
end

function RLBase.extract_experience(t::PPOTrajectory, learner::PPOLearner)
    if isfull(t)
        (
            states = get_trace(t, :state),
            actions = get_trace(t, :action),
            action_log_probs = get_trace(t, :action_log_prob),
            rewards = get_trace(t, :reward),
            terminals = get_trace(t, :terminal),
            states_plus = t[:state],
        )
    else
        nothing
    end
end

function (π::QBasedPolicy{<:PPOLearner})(obs::BatchObs, ::MinimalActionSet)
    probs = π.learner(obs)
    actions = π.explorer(probs)
    actions_log_prob = log.(probs[CartesianIndex.(actions, 1:size(probs, 2))])
    actions, actions_log_prob
end

function (agent::Agent{<:QBasedPolicy{<:PPOLearner}, <:PPOTrajectory})(
    ::PreActStage,
    obs,
)
    action, action_log_prob = agent.policy(obs)
    state = get_state(obs)
    push!(agent.trajectory; state = state, action = action, action_log_prob=action_log_prob)
    update!(agent.policy, agent.trajectory)

    # the main difference is we'd like to flush the buffer after each update!
    if isfull(agent.trajectory)
        empty!(agent.trajectory)
        push!(agent.trajectory; state = state, action = action, action_log_prob=action_log_prob)
    end

    action
end

function (agent::Agent{<:QBasedPolicy{<:PPOLearner}, <:PPOTrajectory})(
    ::PostActStage,
    obs,
)
    push!(agent.trajectory; reward = get_reward(obs), terminal = get_terminal(obs))
    nothing
end