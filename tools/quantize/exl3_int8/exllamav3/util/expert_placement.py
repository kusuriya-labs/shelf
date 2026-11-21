"""CPU-only validation of static expert storage permutations.

An order maps storage slot to original expert ID. Router logits remain in their
original order; the routing kernel translates only the selected output IDs.
"""
import json


def validate_order(order, num_experts):
    if order is None:
        return None
    if not isinstance(order, (list, tuple)) or len(order) != num_experts:
        raise ValueError("Expert order length must match the global expert count")
    if any(type(x) is not int for x in order) or sorted(order) != list(range(num_experts)):
        raise ValueError("Expert order must contain every original expert ID exactly once")
    return tuple(order)


def load_orders(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("version") != 1 or not isinstance(data.get("orders"), dict):
        raise ValueError("Expert placement file requires version=1 and an orders mapping")
    count = data.get("num_experts")
    if type(count) is not int or count <= 0:
        raise ValueError("Expert placement file requires a positive num_experts")
    return {key: validate_order(order, count) for key, order in data["orders"].items()}


def inverse_order(order):
    inverse = [0] * len(order)
    for slot, original in enumerate(order):
        inverse[original] = slot
    return inverse
