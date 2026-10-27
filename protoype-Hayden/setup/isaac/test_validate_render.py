"""Regression checks for the stale-image failure observed during onboarding."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from validate_render import validate


class CaptureChecks(unittest.TestCase):
    def make_pair(self, directory, second_color):
        rows = []
        for i, (color, proxy) in enumerate([('orange', [1, .1, 0]), (second_color, [0, .1, 1])]):
            filename = f'{i}.png'
            Image.new('RGB', (32, 24), color).save(directory / filename)
            rows.append(dict(image_file=filename, linear_rgb_proxy=proxy))
        (directory / 'images.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_duplicate_images_for_different_colors_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_pair(directory, 'orange')
            with self.assertRaisesRegex(ValueError, 'stale'):
                validate(directory)

    def test_visible_color_change_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_pair(directory, 'blue')
            self.assertEqual(validate(directory)['unique_images'], 2)


if __name__ == '__main__':
    unittest.main()
