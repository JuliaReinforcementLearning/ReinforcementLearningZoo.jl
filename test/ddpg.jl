@testset "DDPGPolicy" begin
    inner_env=PendulumEnv(T=Float32,seed=9231)
    action_space = get_action_space(inner_env)
    low = action_space.low
    high = action_space.high
    ns = length(rand(get_observation_space(inner_env)))

    env = WrappedEnv(;
        env = inner_env,
        postprocessor= x -> low + (x + 1) * 0.5 * (high - low) # rescale [-1, 1] -> (low, high)
    );

    agent=Agent(
        policy=DDPGPolicy(
            behavior_approximator=NeuralNetworkApproximator(
                model=ActorCritic(
                    actor = Chain(
                        Dense(ns, 256, relu; initW = seed_glorot_uniform(seed = 17)),
                        Dense(256, 256, relu; initW = seed_glorot_uniform(seed = 154)),
                        Dense(256, 1, tanh; initW = seed_glorot_uniform(seed = 23)),
                    ),
                    critic = Chain(
                        Dense(ns+1, 256, relu; initW = seed_glorot_uniform(seed = 29)),
                        Dense(256, 256, relu; initW = seed_glorot_uniform(seed = 134)),
                        Dense(256, 1; initW = seed_glorot_uniform(seed = 29)),
                    ),
                ),
                optimizer = ADAM(),
                kind = HYBRID_APPROXIMATOR,
            ),
            target_approximator=NeuralNetworkApproximator(
                model=ActorCritic(
                    actor = Chain(
                        Dense(ns, 256, relu; initW = seed_glorot_uniform(seed = 17)),
                        Dense(256, 256, relu; initW = seed_glorot_uniform(seed = 154)),
                        Dense(256, 1, tanh; initW = seed_glorot_uniform(seed = 23)),
                    ),
                    critic = Chain(
                        Dense(ns+1, 256, relu; initW = seed_glorot_uniform(seed = 29)),
                        Dense(256, 256, relu; initW = seed_glorot_uniform(seed = 134)),
                        Dense(256, 1; initW = seed_glorot_uniform(seed = 29)),
                    ),
                ),
                optimizer = ADAM(),
                kind = HYBRID_APPROXIMATOR,
            ),
            γ=0.99f0,
            ρ=0.995f0,
            batch_size=64,
            start_steps=1000,
            start_policy=RandomPolicy(ContinuousSpace(-1.0, 1.0); seed=923),
            update_after=1000,
            update_every=1,
            act_limit=1.0,
            act_noise=0.1,
            seed=131
            ),
        trajectory=CircularCompactSARTSATrajectory(
            capacity = 10000,
            state_type = Float32,
            state_size = (ns,),
            action_type = Float32
        )
    );
    hook = TotalRewardPerEpisode()
    run(agent, env, StopAfterStep(10000),hook)
    @info "stats for DDPGPolicy" avg_reward = mean(hook.rewards)
end