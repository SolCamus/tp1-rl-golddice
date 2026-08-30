import numpy as np

from config import (
    HORIZON,
    DICE_FACES,
    SHIELD_COST,
    get_new_dice_cost,
    get_upgrade_cost,
)

from env import (
    PASS,
    SCORE,
    BUY_DICE,
    UPGRADE,
    BUY_SHIELD,
)


class RandomLegalAgent:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env):
        action = int(self.rng.choice(env.get_valid_actions()))
        score_amount = None

        if action == SCORE:
            score_amount = int(self.rng.choice(env.get_valid_score_amounts()))

        return action, score_amount


class SimpleExpectancyAgent:

    def act(self, obs, env=None):
        turn = obs["turn"]
        gold = obs["gold"]
        num_dice = obs["num_dice"]
        dice_bonus = obs["dice_bonus"]
        shields = obs["shields"]

        turns_left = HORIZON - turn

        if turns_left == 0:
            return SCORE, gold

        if shields == 0 and gold >= SHIELD_COST:
            return BUY_SHIELD, None

        best_action = PASS
        best_value = 0.0

        dice_cost = get_new_dice_cost(num_dice)
        if gold >= dice_cost:
            buy_dice_value = (float(np.mean(DICE_FACES)) + dice_bonus) * turns_left - dice_cost
            if buy_dice_value > best_value:
                best_value = buy_dice_value
                best_action = BUY_DICE

        upgrade_cost = get_upgrade_cost(dice_bonus)
        if gold >= upgrade_cost:
            upgrade_value = num_dice * turns_left - upgrade_cost
            if upgrade_value > best_value:
                best_value = upgrade_value
                best_action = UPGRADE

        return best_action, None


class QLearningAgent:
    """
    Agente entrenado con Q-Learning tabular off-policy (ver train_qlearning.py
    para el detalle del entrenamiento: discretizacion de estado, exploracion
    guiada por SimpleExpectancyAgent + epsilon-greedy, y reward shaping por
    potencial).

    Carga una tabla Q ya entrenada (q_table.pkl, generada por
    train_qlearning.py) y la usa para decidir en cada turno. Si el estado no
    fue visitado durante el entrenamiento (posible dado que el oro no esta
    acotado), cae de forma segura en la heuristica SimpleExpectancyAgent en
    vez de tomar una accion arbitraria.
    """

    def __init__(self, q_table_path="q_table.pkl"):
        import pickle
        from train_qlearning import select_action, tabular_action_to_env

        with open(q_table_path, "rb") as f:
            self.Q = pickle.load(f)

        self._select_action = select_action
        self._tabular_action_to_env = tabular_action_to_env
        self._fallback = SimpleExpectancyAgent()

    def act(self, obs, env):
        tabular_action = self._select_action(self.Q, obs, env, expert_fallback=self._fallback)
        return self._tabular_action_to_env(tabular_action, env.gold)
