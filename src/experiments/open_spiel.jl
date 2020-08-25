function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:Minimax},
    ::Val{:OpenSpiel},
    game;
)
    env = OpenSpielEnv(string(game))
    agents = (
        Agent(policy=MinimaxPolicy(), role=0),
        Agent(policy=MinimaxPolicy(), role=1)
    )
    hooks = (TotalRewardPerEpisode(), TotalRewardPerEpisode())
    description="""
    # Play `$game` in OpenSpiel with Minimax
    """
    Experiment(agents, env, StopAfterEpisode(1), hooks, description)
end