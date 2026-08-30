from evaluate_agents import evaluate
from agents import RandomLegalAgent, SimpleExpectancyAgent, QLearningAgent

Q_TABLE_PATH = "qtable.pkl"


def main():
    agents = {
        "RandomLegal": RandomLegalAgent(seed=123),
        "SimpleExpectancy": SimpleExpectancyAgent(),
        "QLearning": QLearningAgent(q_table_path=Q_TABLE_PATH),
    }

    print(f"{'Agente':<20} {'mean':>8} {'std':>8} {'min':>6} {'p25':>8} {'median':>8} {'p75':>8} {'max':>6}")
    print("-" * 80)

    for name, agent in agents.items():
        result = evaluate(agent, n_episodes=1000, seed=0)
        print(
            f"{name:<20} "
            f"{result['mean']:>8.2f} "
            f"{result['std']:>8.2f} "
            f"{result['min']:>6d} "
            f"{result['p25']:>8.2f} "
            f"{result['median']:>8.2f} "
            f"{result['p75']:>8.2f} "
            f"{result['max']:>6d}"
        )


if __name__ == "__main__":
    main()