
def play_agent(env,agent,agent_name):

    state = env.reset()

    for _ in range(10):

        terminated = False
        while not terminated:
            env.render("human")
            # Take an action and observe next_state, reward and if terminated
            if agent_name != "DQN":
                action, _, _ = agent.take_action(state)
            else:
                agent.eps = 0
                action = agent.take_action(state)
            next_state, reward, terminated, _ = env.step(action)
            state = next_state if not terminated else env.reset()