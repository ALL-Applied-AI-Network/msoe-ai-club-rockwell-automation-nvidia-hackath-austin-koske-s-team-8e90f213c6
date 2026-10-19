from collections import Counter
from copy import deepcopy
import unittest
import subprocess
import sys
from pathlib import Path
import numpy as np

from box_annotations import validate_fixture
from liquid_geometry import body_mesh, stream_mesh
from prepare_pale_review import make


class RevisedGeometryChecks(unittest.TestCase):
    def setUp(self):
        self.fixture=make([dict(source_run=run,image_sha256='a'*64) for run in (1,2,3)])

    def check_solid(self, points, faces):
        p=np.asarray(points); f=np.asarray(faces)
        edges=Counter((int(a),int(b)) for face in f for a,b in zip(face,np.roll(face,-1)))
        self.assertTrue(all(count==1 and edges[(b,a)]==1 for (a,b),count in edges.items()))
        volume=np.einsum('ij,ij->i',p[f[:,0]],np.cross(p[f[:,1]],p[f[:,2]])).sum()/6
        self.assertGreater(volume,0.)
        self.assertTrue(np.isfinite(p).all())

    def test_every_state_has_closed_outward_body_and_stream(self):
        for row in self.fixture['cases']:
            self.check_solid(*body_mesh(row,.0132,.1229)[:2])
            self.check_solid(*stream_mesh(row,1.1776,.9762+row['level']*.1229)[:2])

    def test_first_patch_smaller_than_full_radius_pool_and_meniscus_curved(self):
        rows=self.fixture['cases']
        early,_,detail=body_mesh(rows[4],.0132,.1229)
        full,_,_=body_mesh(rows[8],.0132,.1229)
        self.assertLess(np.ptp(np.array(early)[:,0]),np.ptp(np.array(full)[:,0])*.5)
        self.assertLess(detail['height_m'],.001)
        self.assertGreater(np.ptp(np.array(full)[:641,2]),.0005)

    def test_reference_exclusion_and_invalid_optics_fail_before_kit(self):
        for key,value in [('absorption_depth_m',0),('transmission_color',[1,0,1]),('pool_radius_fraction',2)]:
            wrong=deepcopy(self.fixture);wrong['cases'][0][key]=value
            with self.assertRaises(ValueError):validate_fixture(wrong)
        wrong=deepcopy(self.fixture);wrong['appearance_references'][0]['source_run']=5
        with self.assertRaisesRegex(ValueError,'Runs 1-3'):validate_fixture(wrong)

    def test_revised_preflight_preserves_no_numpy_before_kit(self):
        subprocess.run([sys.executable,'-c',"import sys; import liquid_geometry; "
            "from prepare_pale_review import make; make([dict(source_run=1,image_sha256='a'*64)]); "
            "assert 'numpy' not in sys.modules"],cwd=Path(__file__).parent,check=True)


if __name__=='__main__':unittest.main()
