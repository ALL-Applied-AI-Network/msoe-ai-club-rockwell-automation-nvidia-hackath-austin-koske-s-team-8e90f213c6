import unittest

from replay.recipes_order import library_id, schedule_from_plan

LIBRARY = {"recipes": [
    {"id": "5", "t100_ml": 10.0, "t200_ml": 10.0, "t400_ml": 10.0, "order": "T200>T400>T100"},
    {"id": "7", "t100_ml": 20.0, "t200_ml": 0.0, "t400_ml": 50.0, "order": "T400>T100"},
]}


class RecipeOrderTest(unittest.TestCase):
    def test_exact_id(self):
        self.assertEqual(library_id({"current_recipe_id": "5"}, LIBRARY), "5")

    def test_formula_match_under_another_name(self):
        ordered = {"current_recipe_id": "pale-blue", "t100_ml": 20, "t200_ml": 0, "t300_ml": 0, "t400_ml": 50,
                   "dispense_order": "T400>T100"}
        self.assertEqual(library_id(ordered, LIBRARY), "7")

    def test_same_volumes_different_order_is_not_the_same_recipe(self):
        ordered = {"current_recipe_id": "x", "t100_ml": 10, "t200_ml": 10, "t400_ml": 10, "dispense_order": "T100>T200>T400"}
        self.assertEqual(library_id(ordered, LIBRARY), "x")

    def test_recipe_number(self):
        self.assertEqual(library_id({"current_recipe_id": "recipe-07"}, LIBRARY), "7")

    def test_unknown_keeps_its_id(self):
        self.assertEqual(schedule_from_plan([{"current_recipe_id": "new-teal"}], LIBRARY), ["new-teal"])


if __name__ == "__main__":
    unittest.main()
