"""Map an operator's recipe order (as entered from CSI's machine) onto the recipe library.

A recipe on the line is its formula: millilitres from each tank and the dispense order. The
operator copies that formula into the app, possibly under a different name, so an ordered
vial is matched to a library recipe by, in turn: the same id, the same formula, or the same
recipe number ("recipe-07" -> "7"). An unmatched vial keeps its own id and has no reference.
"""
import re

TANKS = ("t100_ml", "t200_ml", "t300_ml", "t400_ml")


def _formula(recipe):
    volumes = tuple(round(float(recipe.get(k) or 0.0), 2) for k in TANKS)
    order = (recipe.get("dispense_order") or recipe.get("order") or "").replace(" ", "").upper()
    return volumes, order


def library_id(recipe, library):
    """Library recipe id for one ordered vial (a RecipeContext as a dict), or its own id."""
    own = str(recipe.get("current_recipe_id") or "")
    ids = {str(r["id"]) for r in library["recipes"]}
    if own in ids:
        return own
    volumes, order = _formula(recipe)
    if any(volumes):
        for r in library["recipes"]:
            lib_volumes, lib_order = _formula(r)
            if lib_volumes == volumes and (not order or not lib_order or order == lib_order):
                return str(r["id"])
    digits = re.findall(r"\d+", own)
    if digits and str(int(digits[-1])) in ids:
        return str(int(digits[-1]))
    return own


def schedule_from_plan(plan, library):
    """Ordered recipe ids for the line runner from a list of recipe-plan recipes."""
    return [library_id(recipe, library) for recipe in plan]
