from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from train_mixed_detector import sha
from train_sim_only import validate_rows,verify_files,verify_epochs


class SimOnlyChecks(unittest.TestCase):
    def test_partial_fitting_cannot_be_exported_as_the_fixed_final_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'results.csv';path.write_text('epoch,time\n1,1.2\n2,2.3\n')
            verify_epochs(path,2)
            with self.assertRaises(ValueError):verify_epochs(path,12)

    def test_real_rows_real_selection_and_duplicate_identities_are_rejected(self):
        row=dict(source_id='synthetic-a',origin='synthetic',source_run=None,split='train',source_sha256='a',image_sha256='a')
        rows=[row,dict(row,source_id='repeat-a',parent_id='synthetic-a')]
        validate_rows(rows,1,2)
        for key,value in [('origin','real'),('source_run',5),('split','val'),('source_id','repeat-a')]:
            wrong=deepcopy(rows);wrong[0][key]=value
            with self.assertRaises(ValueError):validate_rows(wrong,1,2)
        with self.assertRaises(ValueError):validate_rows(rows,2,2)

    def test_unlisted_images_and_changed_files_cannot_enter_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            data=Path(temporary)
            image=data/'images/train/synthetic-a.png';label=data/'labels/train/synthetic-a.txt'
            image.parent.mkdir(parents=True);label.parent.mkdir(parents=True)
            image.write_bytes(b'fixture-encoded-image');label.write_text('0 .5 .5 .1 .8\n')
            rows=[dict(image='images/train/synthetic-a.png',labels='labels/train/synthetic-a.txt',
                image_sha256=sha(image),actual_label_sha256=sha(label))]
            verify_files(data,rows)
            extra=data/'images/train/unlisted-real-run5.png';extra.write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError,'Unlisted'):verify_files(data,rows)
            extra.unlink();image.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):verify_files(data,rows)


if __name__=='__main__':unittest.main()
