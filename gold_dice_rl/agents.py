import numpy as np
from config import get_new_dice_cost, get_upgrade_cost, SHIELD_COST, STORE_DIE_COST

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
    def __init__(self, q_table_path=None):
        self.Q = {}
        if q_table_path is not None:
            import pickle
            with open(q_table_path, "rb") as f:
                self.Q = pickle.load(f)

    def discretize_state(self, obs):
        gold = obs["gold"]
        can_afford_dice = int(gold >= get_new_dice_cost(obs["num_dice"]))
        can_afford_upgrade = int(gold >= get_upgrade_cost(obs["dice_bonus"]))
        can_afford_shield = int(gold >= SHIELD_COST)
        can_afford_store = int(gold >= STORE_DIE_COST)

        turn_phase = min(obs["turn"] // 10, 2)
        dice_bin = min(obs["num_dice"], 5)
        bonus_bin = min(obs["dice_bonus"], 4)
        shield_bin = min(obs["shields"], 2)
        stored_bin = int(obs["stored_value"] > 0)
        roll_max_bin = obs["roll_max"]

        return (
            turn_phase, can_afford_dice, can_afford_upgrade,
            can_afford_shield, can_afford_store,
            dice_bin, bonus_bin, shield_bin, stored_bin, roll_max_bin,
        )

    def act(self, obs, env):
        state = self.discretize_state(obs)
        valid_actions = env.get_valid_actions()
        q_values = {a: self.Q.get((state, a), 0.0) for a in valid_actions}
        best_action = max(q_values, key=q_values.get)
        score_amount = None
        if best_action == SCORE:
            score_amount = obs["gold"]  # placeholder, todavía pendiente
        return best_action, score_amount