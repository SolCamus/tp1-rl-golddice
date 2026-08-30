import pickle
import numpy as np

from config import HORIZON
from env import GoldDiceEnv, SCORE, BUY_DICE, UPGRADE
from agents import QLearningAgent, SCORE_FRACTIONS


# ---- Hiperparámetros ----
N_EPISODES = 1_000_000
ALPHA = 0.075       # tasa de aprendizaje
GAMMA = 0.99        # factor de descuento
EPS_START = 1.0     # exploración inicial (100% al azar)
EPS_END = 0.05       # exploración mínima
EPS_DECAY_EPISODES = int(N_EPISODES * 0.8) # a lo largo de cuántos episodios decae

SEED = 123
Q_TABLE_PATH = "qtable.pkl"


def epsilon_by_episode(ep):
    frac = min(ep / EPS_DECAY_EPISODES, 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)


INVEST_BONUS_MAX = 30.0

def shaped_reward(env_reward, internal_action, obs):
    """Reward real + bonus por invertir, mayor cuanto mas turnos quedan por delante."""
    bonus = 0.0
    if internal_action in (BUY_DICE, UPGRADE):
        turns_left = HORIZON - obs["turn"]
        bonus = INVEST_BONUS_MAX * (turns_left / HORIZON)
    return env_reward + bonus

def choose_action(agent, obs, env, epsilon, rng):
    """Epsilon-greedy sobre las acciones internas."""
    valid_internal_actions = agent._get_valid_internal_actions(obs, env)

    if rng.random() < epsilon:
        idx = rng.integers(len(valid_internal_actions))
        return valid_internal_actions[idx]  # sin str(), sin pasar por numpy array

    state = agent.discretize_state(obs)
    q_values = {a: agent.Q.get((state, a), 0.0) for a in valid_internal_actions}
    return max(q_values, key=q_values.get)

def train():
    rng = np.random.default_rng(SEED)
    agent = QLearningAgent()  # Q = {} vacía

    episode_returns = []

    for ep in range(N_EPISODES):
        env = GoldDiceEnv(obs_mode="dict", seed=SEED + ep, track_history=False)
        obs = env.reset(seed=SEED + ep)
        done = False
        epsilon = epsilon_by_episode(ep)
        episode_return = 0.0

        while not done:
            state = agent.discretize_state(obs)
            internal_action = choose_action(agent, obs, env, epsilon, rng)
            env_action, score_amount = agent._to_env_action(internal_action, obs)

            next_obs, reward, done, info = env.step(env_action, score_amount=score_amount)
            r = shaped_reward(reward, internal_action, obs)  # <- ahora pasa "obs" (el estado ANTES de actuar)
            episode_return += reward  # para trackear, usamos el reward real

            next_state = agent.discretize_state(next_obs)

            if done:
                target = r
            else:
                next_valid = agent._get_valid_internal_actions(next_obs, env)
                next_q_values = [agent.Q.get((next_state, a), 0.0) for a in next_valid]
                target = r + GAMMA * max(next_q_values)

            old_q = agent.Q.get((state, internal_action), 0.0)
            agent.Q[(state, internal_action)] = old_q + ALPHA * (target - old_q)

            obs = next_obs

        episode_returns.append(episode_return)

        if (ep + 1) % 5000 == 0:
            avg_recent = np.mean(episode_returns[-5000:])
            print(f"Episodio {ep+1}/{N_EPISODES} | eps={epsilon:.3f} | "
                  f"puntaje promedio (últimos 5000)={avg_recent:.2f} | "
                  f"estados en tabla={len(agent.Q)}")

    with open(Q_TABLE_PATH, "wb") as f:
        pickle.dump(agent.Q, f)
    print(f"Tabla Q guardada en {Q_TABLE_PATH} ({len(agent.Q)} entradas)")


if __name__ == "__main__":
    train()