"""CPU-only source exclusion and deterministic sampling checks."""
import unittest
from dataset1_real_transfer import sample_indices,validate_source_rows


class RealTransferTests(unittest.TestCase):
    def row(self,run=4):
        return dict(run_id=run,source_sha256='test-source',boxes={f'G{i:02d}':[1,2,49,50] for i in range(1,31)})

    def test_sparse_run_still_samples_distinct_indices(self):
        self.assertEqual(sample_indices(5),[0,1,3,4])

    def test_run5_refused(self):
        with self.assertRaises(ValueError):validate_source_rows([self.row(5)])

    def test_invalid_crop_refused(self):
        row=self.row();row['boxes']['G01']=[-2,0,46,48]
        with self.assertRaises(ValueError):validate_source_rows([row])

    def test_repeated_source_refused(self):
        with self.assertRaises(ValueError):validate_source_rows([self.row(),self.row()])


if __name__=='__main__':unittest.main()
