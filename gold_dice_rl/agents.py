import numpy as np
from config import (
    HORIZON,
    DICE_FACES,
    SHIELD_COST,
    STORE_DIE_COST,
    get_new_dice_cost,
    get_upgrade_cost,
)

from env import (
    PASS,
    SCORE,
    BUY_DICE,
    UPGRADE,
    BUY_SHIELD,
    STORE_BEST_DIE,   # <- esto faltaba
)

SCORE_FRACTIONS = {
    "SCORE_25": 0.25,
    "SCORE_50": 0.50,
    "SCORE_75": 0.75,
    "SCORE_100": 1.0,
}
TURNS_LEFT_BIN_WIDTH = 3  # bins de a 3 turnos en vez de a 10

INTERNAL_ACTIONS = [
    PASS, "SCORE_25", "SCORE_50", "SCORE_75", "SCORE_100",
    BUY_DICE, UPGRADE, BUY_SHIELD, STORE_BEST_DIE,
]

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
    def __init__(self, q_table_path=None):
        self.Q = {}
        if q_table_path is not None:
            import pickle
            with open(q_table_path, "rb") as f:
                self.Q = pickle.load(f)

    def _bin_roll_max(self, roll_max):
        if roll_max <= 2:
            return 0
        elif roll_max <= 5:
            return 1
        else:
            return 2

    def discretize_state(self, obs):
        gold = obs["gold"]
        turns_left = HORIZON - obs["turn"]

        can_afford_dice = int(gold >= get_new_dice_cost(obs["num_dice"]))
        can_afford_upgrade = int(gold >= get_upgrade_cost(obs["dice_bonus"]))
        can_afford_shield = int(gold >= SHIELD_COST)
        can_afford_store = int(gold >= STORE_DIE_COST)

        turns_left_bin = min(turns_left // TURNS_LEFT_BIN_WIDTH, 9)
        dice_bin = min(obs["num_dice"], 10)
        bonus_bin = min(obs["dice_bonus"], 8)
        shield_bin = min(obs["shields"], 2)
        stored_bin = int(obs["stored_value"] > 0)
        roll_max_bin = self._bin_roll_max(obs["roll_max"])   # <- este es el cambio

        return (
            turns_left_bin, can_afford_dice, can_afford_upgrade,
            can_afford_shield, can_afford_store,
            dice_bin, bonus_bin, shield_bin, stored_bin, roll_max_bin,
        )
    def act(self, obs, env):
        state = self.discretize_state(obs)
        valid_internal_actions = self._get_valid_internal_actions(obs, env)

        q_values = {a: self.Q.get((state, a), 0.0) for a in valid_internal_actions}
        best_internal_action = max(q_values, key=q_values.get)

        return self._to_env_action(best_internal_action, obs)

    def _to_env_action(self, internal_action, obs):
        if internal_action in SCORE_FRACTIONS:
            amount = int(obs["gold"] * SCORE_FRACTIONS[internal_action])
            return SCORE, amount
        return internal_action, None

    def _get_valid_internal_actions(self, obs, env):
        valid = env.get_valid_actions()
        actions = [a for a in valid if a != SCORE]  # las no-SCORE se copian tal cual
        if obs["gold"] > 0:  # solo ofrecer variantes de SCORE si hay algo para puntuar
            actions += list(SCORE_FRACTIONS.keys())
        return actions