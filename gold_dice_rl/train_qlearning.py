from env import GoldDiceEnv
from agents import QLearningAgent

agent = QLearningAgent()  # arranca con Q vacía, sin cargar tabla

# acá va el loop de entrenamiento, usando agent.discretize_state(obs)
# para indexar la tabla agent.Q y actualizarla con la regla de Q-Learning