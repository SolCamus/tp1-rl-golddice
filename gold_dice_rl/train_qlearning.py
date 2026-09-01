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

Mejoras sobre la primera version (que llegaba a ~470-475 puntos, ver commit
acd4dd9). Resultado final: >535 puntos promedio sobre 1000 episodios
(seed=0), vs ~347 del heuristico SimpleExpectancy y ~470 de la primera
version.

Incorporadas y activas por defecto:

1. Exploracion softmax (Boltzmann) en vez de epsilon-random uniforme: al
   explorar, en vez de tirar una accion legal al azar, se samplea
   proporcional a exp(Q/temperatura_efectiva) entre las acciones legales
   (ver `_softmax_choice`). Explora mucho entre acciones parecidas (que es
   donde realmente hace falta explorar) y casi no pierde episodios
   probando una accion claramente mala que ya se sabe que es mala.
   OJO: la primera version de esta idea tenia un bug serio que la dejaba
   sin efecto util (ver mas abajo) — el resultado de >535 es CON el bug
   arreglado.
2. Potencial de reward shaping mas rico: Phi(s) ya no es solo `gold`, sino
   que penaliza tener oro expuesto sin escudo (ver `_potential`). Sigue
   siendo shaping basado en potencial (Ng, Harada & Russell, 1999), asi que
   no cambia la politica optima, pero densifica la señal justo alrededor de
   la decision de comprar escudo.
3. Loop de entrenamiento optimizado (mismo resultado, ~2.2x mas rapido):
   se reutiliza un unico GoldDiceEnv entre episodios (env.reset() por
   episodio en vez de instanciar uno nuevo, que ya hacia su propio reset
   internamente) y se evita recalcular la mascara de acciones validas dos
   veces por paso. Verificado bit-a-bit contra la version anterior (mismo
   seed, mismo resultado exacto: 470.584).

Bug encontrado y corregido (el hallazgo mas importante de esta ronda):
la primera implementacion de la exploracion softmax dividia Q por una
`temperature` absoluta pensada para valores chicos (0.1-1.0), pero los Q
de este problema son sumas de puntaje sobre hasta 30 turnos (~10^2-10^3).
exp(Q/temperature) colapsaba casi siempre al argmax exacto sin que se
notara: la rama de "exploracion" terminaba actuando casi igual que la
rama greedy, es decir, se perdia casi toda la exploracion real. Con eso
el resultado quedaba plantado en ~358-360 sin importar otros cambios
(se probo con topes de estado viejos Y nuevos, mismo resultado en ambos
casos, lo cual en su momento hizo sospechar erroneamente de los topes).
La correccion: `temperature` ahora es relativa al spread real de Q entre
las acciones validas de cada estado (no un valor absoluto), asi el
comportamiento no depende de la escala de Q. Una vez arreglado, la misma
receta subio de ~360 a >535.

Probadas y DESCARTADAS (documentado para no repetir el esfuerzo):

- Topes de discretizacion mas altos (MAX_NUM_DICE/MAX_DICE_BONUS/
  MAX_SHIELDS subidos de 6/5/1 a 10/8/4): la hipotesis era que el agente
  se "achataba" contra el techo del estado en 28%/27%/62% de los turnos
  jugados (medido con inspect_qtable.py) y perdia informacion en las
  partidas de alta inversion. Con 7.1M episodios (~3x el presupuesto
  original) el resultado quedo plantado en ~360 — pero esa prueba se hizo
  ANTES de encontrar el bug de softmax de arriba, asi que esta conclusion
  esta confundida con ese bug (con exploracion realmente rota, ni los
  topes viejos pasaban de ~358 tampoco). No se volvio a probar con el
  bug ya arreglado por falta de tiempo: sigue siendo candidata valida si
  se busca superar el resultado actual.
- Double Q-Learning (dos tablas, target cruzado, para reducir el sesgo de
  sobreestimacion): interactuaba mal con el alpha adaptativo por visitas
  (cada tabla recibe la mitad de las actualizaciones reales pero alpha
  decae contando las visitas de ambas, así que decae al doble de rapido
  de lo que le corresponde) y el resultado empeoraba en vez de mejorar.
  Tampoco se reintento tras el fix de softmax.
"""

import math
import pickle
import random
from collections import defaultdict

import numpy as np

from config import HORIZON, STORM_PROB
from env import GoldDiceEnv, PASS, SCORE, BUY_DICE, UPGRADE, BUY_SHIELD, STORE_BEST_DIE
from agents import SimpleExpectancyAgent

# --------------------------------------------------------------------------
# Discretizacion de estado
# --------------------------------------------------------------------------
#
# turns_left: precision fina cerca del final del horizonte (ahi importa el
#   timing exacto), gruesa lejos del final (ahi lo que importa es "hay
#   tiempo para amortizar una inversion", no el turno exacto).
# gold: variable no acotada -> se discretiza en buckets alineados a costos.
# num_dice / dice_bonus: se capean (valores muy altos son raros y se
#   comportan igual a fines practicos).
# shields: solo importa si el jugador tiene 0 o al menos 1 (para decidir si
#   comprar otro no aporta demasiado, se capea en 1).
#
# Se probo subir estos topes (MAX_NUM_DICE/MAX_DICE_BONUS/MAX_SHIELDS a
# 10/8/4) para que el agente no se quedara "ciego" en partidas de alta
# inversion, pero se descarto: el juego casi nunca llega a esos niveles,
# asi que la resolucion extra solo diluia datos en la zona comun sin mejorar
# el resultado (ver docstring del modulo).
#
# roll_sum / roll_max del obs original se dejan afuera del estado: solo
# afectan la decision de STORE_BEST_DIE, una optimizacion menor, y
# agregarlas multiplicaba el espacio de estados sin aportar mucho (probado,
# ver notas de experimentos).

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
    """Funcion de potencial para reward shaping.

    Reward shaping basado en potencial (Ng, Harada & Russell, 1999):
        r' = r + gamma * Phi(s') - Phi(s)
    Con Phi(terminal)=0, la suma de rewards shaped de un episodio coincide
    exactamente con la suma de rewards originales (el puntaje final), por lo
    que Phi(s) puede ser CUALQUIER funcion del estado sin cambiar cual es la
    politica optima: el shaping solo afecta que tan rapido se aprende, no
    que se aprende.

    Phi(s) = oro esperado "seguro": el oro en mano vale su valor pleno si
    hay al menos un escudo, pero si shields=0 se descuenta la probabilidad
    de tormenta (que parte el oro a la mitad). Esto densifica la señal
    justo alrededor de BUY_SHIELD: antes de comprar escudo con mucho oro
    en juego, Phi(s) ya es mas bajo que el oro nominal, y comprar el escudo
    hace que Phi salte para arriba de inmediato (en vez de que el agente
    tenga que descubrir el valor del escudo solo a partir de tormentas que
    caen 15% de las veces, señal muy dispersa).
    """
    gold = float(obs["gold"])
    if obs["shields"] <= 0:
        # oro esperado tras la tormenta de este turno si no se protege.
        return gold * (1.0 - STORM_PROB) + gold * STORM_PROB * 0.5
    return gold


# --------------------------------------------------------------------------
# Exploracion softmax (Boltzmann)
# --------------------------------------------------------------------------


def _softmax_choice(rng, q_values, valid_actions, temperature):
    """Elige una accion legal muestreando de softmax(Q/temperature_efectiva).

    Reemplaza la exploracion epsilon-random uniforme: en vez de tirar
    cualquier accion legal con la misma probabilidad (incluidas las que ya
    se sabe que son claramente malas), pondera por que tan buena parece
    cada accion segun la Q actual. A temperatura alta se comporta casi
    como uniforme (mucha exploracion); a temperatura baja se acerca al
    argmax (poca exploracion, pero nunca determinista del todo).

    OJO con la escala: los Q de este problema son sumas de puntaje sobre
    hasta 30 turnos, tipicamente de orden ~10^2-10^3, no ~1. Una
    `temperature` absoluta pensada para valores chicos (p.ej. 0.1-1.0)
    hace que exp(Q/temperature) colapse casi siempre al argmax exacto, es
    decir, se pierde casi toda la exploracion sin que se note (asi fallo
    la primera version de esta funcion: la rama de "exploracion" terminaba
    actuando practicamente igual que la rama greedy). Por eso `temperature`
    se interpreta ACA como una fraccion relativa al spread real de Q entre
    las acciones validas en ESE estado (no un valor absoluto), asi el
    comportamiento no depende de la escala de Q.
    """
    vals = [q_values[a] for a in valid_actions]
    spread = max(vals) - min(vals)

    if temperature <= 1e-6:
        return max(valid_actions, key=lambda a: q_values[a])
    if spread <= 1e-9:
        # Q practicamente empatada entre las acciones validas: no hay nada
        # que pesar, cualquiera sirve para explorar.
        return rng.choice(valid_actions)

    scale = temperature * spread
    m = max(vals)
    weights = [math.exp((v - m) / scale) for v in vals]
    total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        return rng.choice(valid_actions)

    u = rng.random() * total
    cum = 0.0
    for a, w in zip(valid_actions, weights):
        cum += w
        if u <= cum:
            return a
    return valid_actions[-1]


# --------------------------------------------------------------------------
# Entrenamiento Q-Learning
# --------------------------------------------------------------------------
#
# Nota sobre Double Q-Learning: se probo (dos tablas QA/QB, target cruzado)
# y se descarto. Con un solo contador de visitas compartido para calcular
# alpha, cada tabla recibe la mitad de las actualizaciones reales pero
# alpha decae a la misma velocidad que en Q-Learning simple: combinado con
# el alpha_min bajo del schedule (pensado para actualizaciones 1:1), las
# tablas quedaban practicamente congeladas apenas se soltaba el heuristico
# (avg score de entrenamiento se planchaba en ~340 en vez de subir), y el
# resultado final (348) no superaba al heuristico. Se podria arreglar
# llevando un contador de visitas por tabla, pero el sesgo de
# sobreestimacion no parecia ser el cuello de botella real de este
# problema, asi que no se le dedico mas tiempo.


def train_q_learning(
    n_episodes=1_000_000,
    gamma=1.0,
    alpha_min=0.01,
    epsilon=0.08,
    temperature=0.5,
    seed=0,
    log_every=20_000,
    use_shaping=True,
    p_expert=0.7,
    Q_init=None,
    visits_init=None,
    episode_offset=0,
):
    """
    Politica de comportamiento mixta: con prob. p_expert imita al
    heuristico SimpleExpectancyAgent (exploracion guiada), si no con prob.
    epsilon explora con softmax sobre Q (ver `_softmax_choice`), si no
    actua greedy sobre Q.

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
    # Un solo GoldDiceEnv reutilizado entre episodios (reset() por episodio)
    # en vez de instanciar uno nuevo cada vez: GoldDiceEnv.__init__ ya llama
    # a reset() internamente, asi que crear el objeto Y llamar reset() de
    # nuevo (como hacia la version anterior) duplicaba la creacion del RNG
    # y la tirada inicial de dados sin necesidad.
    env = GoldDiceEnv(obs_mode="dict", track_history=False)

    for ep in range(n_episodes):
        ep_seed = seed * 10_000_000 + episode_offset + ep
        obs = env.reset(seed=ep_seed)
        state = discretize_state(obs)
        valid_actions = valid_tabular_actions(env)
        done = False

        while not done:
            if rng.random() < p_expert:
                heur_action, heur_amount = expert.act(obs, env)
                action = _heuristic_to_tabular_action(heur_action, heur_amount, env.gold)
                if action not in valid_actions:
                    action = ACTION_PASS
            elif rng.random() < epsilon:
                action = _softmax_choice(rng, Q[state], valid_actions, temperature)
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
                next_valid = None
            else:
                # valid_tabular_actions(env) ya refleja el estado post-accion
                # (s'): sirve de una para el bootstrap del target Y como
                # valid_actions de la PROXIMA iteracion del while (evita
                # recalcular la mascara de acciones dos veces por paso).
                next_valid = valid_tabular_actions(env)
                target = learn_reward + gamma * max(Q[next_state][a] for a in next_valid)

            visits[state][action] += 1
            alpha = max(alpha_min, 1.0 / (1.0 + visits[state][action]))
            Q[state][action] += alpha * (target - Q[state][action])

            state = next_state
            obs = next_obs
            valid_actions = next_valid

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
    # heuristico por su cuenta. La temperatura de la exploracion softmax
    # tambien baja con el tiempo (mas exploracion amplia al principio, mas
    # afinada al final). Cada tanda continua desde la Q de la anterior
    # (Q_init/visits_init), no arranca de cero.
    # Presupuesto ~2.7x el original (6.4M vs 2.4M episodios): con los
    # topes de estado viejos (probados, no diluyen datos como los topes
    # altos que se descartaron) y el loop ~2.2x mas rapido, esto corre en
    # tiempo parecido al original pero le da mucho mas datos al potencial
    # de riesgo y a la exploracion softmax (ver `_softmax_choice`) para
    # asentarse. Las ultimas 4 tandas son una fase de refinamiento fino
    # (epsilon y temperatura bajando mas, siempre p_expert=0): el avg
    # score de entrenamiento seguia subiendo sin aplanarse del todo al
    # llegar a la tanda 13, asi que se la siguio un poco mas.
    #
    # Con esta receta se llega a >535 puntos promedio sobre 1000 episodios
    # (seed=0), vs ~347 del heuristico SimpleExpectancy y ~470 de la
    # primera version de este agente.
    schedule = [
        # (n_episodios, epsilon, p_expert, alpha_min, temperature)
        (300_000, 0.08, 0.70, 0.010, 1.0),
        (300_000, 0.08, 0.70, 0.008, 1.0),
        (300_000, 0.08, 0.70, 0.006, 0.8),
        (300_000, 0.08, 0.70, 0.005, 0.8),
        (300_000, 0.10, 0.55, 0.004, 0.6),
        (300_000, 0.12, 0.40, 0.003, 0.6),
        (300_000, 0.15, 0.25, 0.0025, 0.5),
        (300_000, 0.18, 0.15, 0.002, 0.4),
        (300_000, 0.20, 0.08, 0.0015, 0.3),
        (300_000, 0.22, 0.03, 0.0012, 0.25),
        (400_000, 0.22, 0.00, 0.0012, 0.2),
        (400_000, 0.18, 0.00, 0.001, 0.15),
        (400_000, 0.15, 0.00, 0.001, 0.1),
        (500_000, 0.15, 0.00, 0.001, 0.1),
        (500_000, 0.12, 0.00, 0.0008, 0.08),
        (500_000, 0.10, 0.00, 0.0006, 0.06),
        (700_000, 0.08, 0.00, 0.0005, 0.05),
    ]

    Q, visits = None, None
    episode_offset = 0

    for i, (n_ep, eps, p_exp, a_min, temp) in enumerate(schedule, start=1):
        print(f"\n=== Tanda {i}/{len(schedule)}: "
              f"n_episodes={n_ep} epsilon={eps} p_expert={p_exp} "
              f"alpha_min={a_min} temperature={temp} ===")
        Q, visits = train_q_learning(
            n_episodes=n_ep,
            epsilon=eps,
            p_expert=p_exp,
            alpha_min=a_min,
            temperature=temp,
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
