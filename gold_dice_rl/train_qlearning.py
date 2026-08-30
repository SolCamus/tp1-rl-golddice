"""
Entrenamiento de un agente Q-Learning tabular para Gold Dice RL.

Enfoque:
- El estado (obs) tiene variables no acotadas (gold, en particular), asi que
  se discretiza en buckets para poder usar una tabla Q (dict).
- La accion SCORE requiere un monto (score_amount). Para mantener el problema
  tabular, se discretiza en un conjunto chico de acciones combinadas:
  PASS, SCORE_ALL, BUY_DICE, UPGRADE, BUY_SHIELD, STORE_BEST_DIE.
  (No se incluye "SCORE parcial": la estrategia optima de este juego es
  invertir el oro en producir mas oro y puntuar recien al final, nunca
  puntuar a medias tiene sentido salvo casos borde muy raros, y el
  heuristico provisto tampoco lo usa.)
- gamma = 1.0: el episodio es de horizonte finito (30 turnos) y la reward
  que devuelve env.step ya es exactamente el score_amount, por lo que la
  suma de rewards de un episodio es igual al puntaje final. No hace falta
  descontar.
- El estado se mantiene chico a proposito (5 variables, todas con pocos
  valores posibles) para que la tabla Q tenga buena cobertura con una
  cantidad de episodios manejable en una laptop.
"""

import pickle
import random
from collections import defaultdict

import numpy as np

from config import HORIZON
from env import GoldDiceEnv, PASS, SCORE, BUY_DICE, UPGRADE, BUY_SHIELD, STORE_BEST_DIE
from agents import SimpleExpectancyAgent

# --------------------------------------------------------------------------
# Discretizacion de estado
# --------------------------------------------------------------------------
#
# turns_left: precision fina cerca del final del horizonte (ahi importa el
#   timing exacto), gruesa lejos del final (ahi lo que importa es "hay
#   tiempo para amortizar una inversion", no el turno exacto).
# gold: variable no acotada -> se discretiza en buckets log-espaciados.
# num_dice / dice_bonus: se capean (valores muy altos son raros y se
#   comportan igual a fines practicos).
# shields: solo importa si el jugador tiene 0 o al menos 1 (para decidir si
#   comprar otro no aporta demasiado, se capea en 1).
#
# roll_sum / roll_max del obs original se dejan afuera del estado: solo
# afectan la decision de STORE_BEST_DIE, una optimizacion menor, y
# agregarlas multiplicaba el espacio de estados sin aportar mucho.

TURNS_LEFT_EDGES = list(range(0, 30))  # sin bucketizar: el heuristico usa
# turns_left exacto en su formula de valor esperado, asi que agruparlo
# perdia justo la precision que hace falta cerca de los umbrales de costo.
GOLD_EDGES = sorted(set([
    4, 5, 8, 16, 18, 24, 26, 32, 34, 40, 42, 48, 50, 56, 58, 66, 74, 82, 90,
    110, 140, 180, 230, 300, 400, 550,
]))  # alineados a los costos reales de compra (dado, upgrade, escudo) para
# no perder precision justo en los umbrales donde importa la decision.
MAX_NUM_DICE = 6
MAX_DICE_BONUS = 5
MAX_SHIELDS = 1


def _bucket(value, edges):
    idx = 0
    for e in edges:
        if value > e:
            idx += 1
        else:
            break
    return idx


def discretize_state(obs):
    turns_left = HORIZON - int(obs["turn"])
    return (
        _bucket(turns_left, TURNS_LEFT_EDGES),
        _bucket(obs["gold"], GOLD_EDGES),
        min(int(obs["num_dice"]), MAX_NUM_DICE),
        min(int(obs["dice_bonus"]), MAX_DICE_BONUS),
        min(int(obs["shields"]), MAX_SHIELDS),
    )


# --------------------------------------------------------------------------
# Acciones combinadas (accion tabular -> (accion del env, score_amount))
# --------------------------------------------------------------------------

ACTION_PASS = 0
ACTION_SCORE_ALL = 1
ACTION_BUY_DICE = 2
ACTION_UPGRADE = 3
ACTION_BUY_SHIELD = 4
ACTION_STORE_BEST_DIE = 5

N_TABULAR_ACTIONS = 6

# A que accion base del env corresponde cada accion tabular (para chequear
# validez con env.get_action_mask(), que es API publica del ambiente).
_BASE_ACTION_OF = {
    ACTION_PASS: PASS,
    ACTION_SCORE_ALL: SCORE,
    ACTION_BUY_DICE: BUY_DICE,
    ACTION_UPGRADE: UPGRADE,
    ACTION_BUY_SHIELD: BUY_SHIELD,
    ACTION_STORE_BEST_DIE: STORE_BEST_DIE,
}


def valid_tabular_actions(env):
    mask = env.get_action_mask()
    return [a for a, base in _BASE_ACTION_OF.items() if mask[base]]


def tabular_action_to_env(action, gold):
    if action == ACTION_PASS:
        return PASS, None
    if action == ACTION_SCORE_ALL:
        return SCORE, int(gold)
    if action == ACTION_BUY_DICE:
        return BUY_DICE, None
    if action == ACTION_UPGRADE:
        return UPGRADE, None
    if action == ACTION_BUY_SHIELD:
        return BUY_SHIELD, None
    if action == ACTION_STORE_BEST_DIE:
        return STORE_BEST_DIE, None
    raise ValueError(f"Unknown tabular action: {action}")


def _heuristic_to_tabular_action(env_action, score_amount, gold):
    """Traduce la decision del agente heuristico (SimpleExpectancyAgent) a
    nuestro espacio de acciones tabular, para poder usarlo como politica de
    comportamiento (exploracion guiada) durante el entrenamiento off-policy.
    """
    if env_action == PASS:
        return ACTION_PASS
    if env_action == SCORE:
        return ACTION_SCORE_ALL if (score_amount and score_amount > 0) else ACTION_PASS
    if env_action == BUY_DICE:
        return ACTION_BUY_DICE
    if env_action == UPGRADE:
        return ACTION_UPGRADE
    if env_action == BUY_SHIELD:
        return ACTION_BUY_SHIELD
    if env_action == STORE_BEST_DIE:
        return ACTION_STORE_BEST_DIE
    return ACTION_PASS


def _potential(obs):
    """Funcion de potencial para reward shaping: Phi(s) = gold acumulado.

    Reward shaping basado en potencial (Ng, Harada & Russell, 1999):
        r' = r + gamma * Phi(s') - Phi(s)
    Con Phi(terminal)=0 y Phi(estado inicial)=0, la suma de rewards shaped
    de un episodio coincide exactamente con la suma de rewards originales
    (el puntaje final), por lo que no cambia lo que es optimo. Lo que si
    hace es densificar la señal: comprar un dado no da reward inmediata,
    pero el oro extra que ese dado genera en turnos siguientes se refleja
    de inmediato via el cambio de potencial, en vez de esperar a que el
    agente puntue mucho despues.
    """
    return float(obs["gold"])


# --------------------------------------------------------------------------
# Entrenamiento Q-Learning
# --------------------------------------------------------------------------


def train_q_learning(
    n_episodes=1_000_000,
    gamma=1.0,
    alpha_min=0.01,
    epsilon=0.08,
    seed=0,
    log_every=20_000,
    use_shaping=True,
    p_expert=0.7,
    Q_init=None,
    visits_init=None,
    episode_offset=0,
):
    """
    ...
    Q_init / visits_init: si se pasan (dicts de una corrida anterior), el
    entrenamiento continua desde ahi en vez de arrancar de cero (util para
    extender el entrenamiento en varias tandas sin perder lo aprendido).
    episode_offset: para que las semillas de las partidas de esta tanda no
    se repitan con las de tandas anteriores.
    """
    rng = random.Random(seed + episode_offset)
    Q = defaultdict(lambda: np.zeros(N_TABULAR_ACTIONS, dtype=np.float64))
    visits = defaultdict(lambda: np.zeros(N_TABULAR_ACTIONS, dtype=np.int64))
    if Q_init is not None:
        for k, v in Q_init.items():
            Q[k] = np.array(v, dtype=np.float64)
    if visits_init is not None:
        for k, v in visits_init.items():
            visits[k] = np.array(v, dtype=np.int64)
    expert = SimpleExpectancyAgent()

    recent_scores = []

    for ep in range(n_episodes):
        ep_seed = seed * 10_000_000 + episode_offset + ep
        env = GoldDiceEnv(obs_mode="dict", seed=ep_seed, track_history=False)
        obs = env.reset(seed=ep_seed)
        state = discretize_state(obs)
        done = False

        while not done:
            valid_actions = valid_tabular_actions(env)

            if rng.random() < p_expert:
                heur_action, heur_amount = expert.act(obs, env)
                action = _heuristic_to_tabular_action(heur_action, heur_amount, env.gold)
                if action not in valid_actions:
                    action = ACTION_PASS
            elif rng.random() < epsilon:
                action = rng.choice(valid_actions)
            else:
                q_values = Q[state]
                action = max(valid_actions, key=lambda a: q_values[a])

            env_action, score_amount = tabular_action_to_env(action, env.gold)
            phi_s = _potential(obs)
            next_obs, reward, done, info = env.step(env_action, score_amount=score_amount)
            next_state = discretize_state(next_obs)

            if use_shaping:
                phi_next = 0.0 if done else _potential(next_obs)
                learn_reward = reward + gamma * phi_next - phi_s
            else:
                learn_reward = reward

            if done:
                target = learn_reward
            else:
                next_valid = valid_tabular_actions(env)
                target = learn_reward + gamma * max(Q[next_state][a] for a in next_valid)

            visits[state][action] += 1
            alpha = max(alpha_min, 1.0 / (1.0 + visits[state][action]))
            Q[state][action] += alpha * (target - Q[state][action])

            state = next_state
            obs = next_obs

        recent_scores.append(env.points)
        if (ep + 1) % log_every == 0:
            avg = np.mean(recent_scores[-log_every:])
            print(f"Episode {ep + 1}/{n_episodes} | avg score (last {log_every}): {avg:.1f} "
                  f"| states visited: {len(Q)}")

    return dict(Q), dict(visits)


def select_action(Q, obs, env, expert_fallback=None):
    """
    Elige una accion tabular a partir de la tabla Q, con dos salvavidas
    para estados no visitados durante el entrenamiento (inevitable en un
    problema con gold no acotado):

      1. En el ultimo turno (turns_left == 0) siempre puntuar todo el oro.
         Es la accion optima por definicion (no queda nada que hacer con
         el oro despues), no hace falta que la tabla Q lo haya aprendido.
      2. Si el estado no esta en la tabla Q, en vez de caer en una accion
         arbitraria (p.ej. PASS, que tira el turno a la basura), se usa la
         heuristica SimpleExpectancyAgent como politica de resguardo.
    """
    valid = valid_tabular_actions(env)

    turns_left = HORIZON - int(obs["turn"])
    if turns_left == 0:
        return ACTION_SCORE_ALL if ACTION_SCORE_ALL in valid else valid[0]

    state = discretize_state(obs)
    q_values = Q.get(state)

    if q_values is not None:
        return max(valid, key=lambda a: q_values[a])

    if expert_fallback is not None:
        heur_action, heur_amount = expert_fallback.act(obs, env)
        action = _heuristic_to_tabular_action(heur_action, heur_amount, env.gold)
        if action in valid:
            return action

    return valid[0]


if __name__ == "__main__":
    # Schedule de entrenamiento: arranca imitando casi siempre al heuristico
    # (p_expert alto) para evitar el problema de exploracion dispersa, y lo
    # va soltando de a poco (p_expert baja, epsilon sube) a medida que la
    # tabla Q ya tiene una base solida y puede empezar a superar al
    # heuristico por su cuenta. Cada tanda continua desde la Q de la
    # anterior (Q_init/visits_init), no arranca de cero.
    #
    # Con esta receta se llega a ~474 puntos promedio (vs ~347 del
    # heuristico SimpleExpectancy), entrenando en total 2.4M episodios.
    # En una laptop, esto tarda entre 30 y 50 minutos en total (es todo
    # tabular, no hace falta GPU). Se imprime el progreso tanda por tanda.
    schedule = [
        # (n_episodios, epsilon, p_expert, alpha_min)
        (200_000, 0.08, 0.70, 0.010),
        (200_000, 0.08, 0.70, 0.008),
        (200_000, 0.08, 0.70, 0.006),
        (200_000, 0.08, 0.70, 0.005),
        (200_000, 0.08, 0.70, 0.004),
        (200_000, 0.08, 0.70, 0.003),
        (150_000, 0.15, 0.40, 0.003),
        (150_000, 0.15, 0.40, 0.002),
        (150_000, 0.18, 0.25, 0.002),
        (150_000, 0.20, 0.15, 0.0015),
        (150_000, 0.22, 0.08, 0.001),
        (150_000, 0.25, 0.03, 0.001),
        (150_000, 0.25, 0.00, 0.001),
        (150_000, 0.20, 0.00, 0.0008),
    ]

    Q, visits = None, None
    episode_offset = 0

    for i, (n_ep, eps, p_exp, a_min) in enumerate(schedule, start=1):
        print(f"\n=== Tanda {i}/{len(schedule)}: "
              f"n_episodes={n_ep} epsilon={eps} p_expert={p_exp} alpha_min={a_min} ===")
        Q, visits = train_q_learning(
            n_episodes=n_ep,
            epsilon=eps,
            p_expert=p_exp,
            alpha_min=a_min,
            Q_init=Q,
            visits_init=visits,
            episode_offset=episode_offset,
            log_every=n_ep,
        )
        episode_offset += n_ep

        # Guardamos despues de cada tanda: si se corta a mitad de camino,
        # no se pierde el progreso.
        with open("q_table.pkl", "wb") as f:
            pickle.dump(Q, f)

    print(f"\nEntrenamiento terminado. Guardado q_table.pkl con {len(Q)} estados.")
    print("Corre 'python evaluate_agents.py' para ver el resultado final.")
