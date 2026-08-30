import pickle
from collections import defaultdict

Q_TABLE_PATH = "qtable.pkl"

ACTION_NAMES = {
    0: "PASS",
    1: "SCORE",  # no deberia aparecer solo (SCORE se desdobla en SCORE_25/50/75/100)
    2: "BUY_DICE",
    3: "UPGRADE",
    4: "BUY_SHIELD",
    5: "STORE_BEST_DIE",
}


def action_label(action):
    if isinstance(action, str):
        return action
    return ACTION_NAMES.get(action, f"ACCION_DESCONOCIDA({action})")


def describe_state(state):
    (turns_left_bin, can_afford_dice, can_afford_upgrade,
     can_afford_shield, can_afford_store,
     dice_bin, bonus_bin, shield_bin, stored_bin, roll_max_bin) = state

    return (
        f"turns_left_bin={turns_left_bin} | "
        f"gold_alcanza[dado={bool(can_afford_dice)}, "
        f"upg={bool(can_afford_upgrade)}, "
        f"escudo={bool(can_afford_shield)}, "
        f"guardar={bool(can_afford_store)}] | "
        f"dice_bin={dice_bin} | bonus_bin={bonus_bin} | "
        f"shields_bin={shield_bin} | stored={bool(stored_bin)} | "
        f"roll_max_bin={roll_max_bin}"
    )


def load_by_state():
    with open(Q_TABLE_PATH, "rb") as f:
        Q = pickle.load(f)

    by_state = defaultdict(dict)
    for (state, action), value in Q.items():
        by_state[state][action_label(action)] = value

    return Q, by_state


def section_top_states(by_state, n=15):
    print("=" * 90)
    print(f"TOP {n} ESTADOS DE MAYOR VALOR")
    print("=" * 90)

    states_sorted = sorted(
        by_state.items(), key=lambda item: max(item[1].values()), reverse=True,
    )

    for state, action_values in states_sorted[:n]:
        best_action = max(action_values, key=action_values.get)
        best_value = action_values[best_action]
        others = {a: round(v, 1) for a, v in sorted(
            action_values.items(), key=lambda x: -x[1]) if a != best_action}

        print(describe_state(state))
        print(f"  -> Mejor accion: {best_action} (Q={best_value:.2f})")
        if others:
            print(f"     Otras opciones: {others}")
        print()


def section_action_distribution(by_state):
    print("=" * 90)
    print("DISTRIBUCION GLOBAL: cuantas veces cada accion es la 'mejor'")
    print("=" * 90)

    counts = defaultdict(int)
    for state, action_values in by_state.items():
        best_action = max(action_values, key=action_values.get)
        counts[best_action] += 1

    total = len(by_state)
    for action, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"  {action:15s} : {count:5d} estados ({pct:5.1f}%)")
    print()


def section_max_bins_reached(by_state):
    print("=" * 90)
    print("HASTA DONDE LLEGA EL AGENTE: valores maximos de dice_bin y bonus_bin en la tabla")
    print("=" * 90)

    max_dice_bin = 0
    max_bonus_bin = 0
    dice_bin_counts = defaultdict(int)
    bonus_bin_counts = defaultdict(int)

    for state in by_state:
        dice_bin = state[5]
        bonus_bin = state[6]
        max_dice_bin = max(max_dice_bin, dice_bin)
        max_bonus_bin = max(max_bonus_bin, bonus_bin)
        dice_bin_counts[dice_bin] += 1
        bonus_bin_counts[bonus_bin] += 1

    print(f"dice_bin maximo alcanzado: {max_dice_bin}  (cap configurado: 10)")
    print(f"bonus_bin maximo alcanzado: {max_bonus_bin}  (cap configurado: 8)")
    print()

    print("Distribucion de estados por dice_bin:")
    for db in sorted(dice_bin_counts):
        print(f"  dice_bin={db}: {dice_bin_counts[db]} estados")
    print()

    print("Distribucion de estados por bonus_bin:")
    for bb in sorted(bonus_bin_counts):
        print(f"  bonus_bin={bb}: {bonus_bin_counts[bb]} estados")
    print()


def section_invest_actions_by_turns_left(by_state):
    print("=" * 90)
    print("BUY_DICE / UPGRADE como mejor accion, agrupado por turns_left_bin")
    print("(turns_left_bin=0 es el final de la partida, 9 es el principio)")
    print("=" * 90)

    counts_by_bin = defaultdict(lambda: defaultdict(int))
    for state, action_values in by_state.items():
        turns_left_bin = state[0]
        best_action = max(action_values, key=action_values.get)
        counts_by_bin[turns_left_bin][best_action] += 1

    for tlb in sorted(counts_by_bin, reverse=True):
        total_in_bin = sum(counts_by_bin[tlb].values())
        buy_dice = counts_by_bin[tlb].get("BUY_DICE", 0)
        upgrade = counts_by_bin[tlb].get("UPGRADE", 0)
        print(f"  turns_left_bin={tlb} ({total_in_bin} estados): "
              f"BUY_DICE={buy_dice} ({100*buy_dice/total_in_bin:.1f}%), "
              f"UPGRADE={upgrade} ({100*upgrade/total_in_bin:.1f}%)")
    print()


def main():
    Q, by_state = load_by_state()
    print(f"Total de estados distintos: {len(by_state)}")
    print(f"Total de entradas (estado,accion): {len(Q)}\n")

    section_top_states(by_state, n=15)
    section_action_distribution(by_state)
    section_max_bins_reached(by_state)
    section_invest_actions_by_turns_left(by_state)


if __name__ == "__main__":
    main()