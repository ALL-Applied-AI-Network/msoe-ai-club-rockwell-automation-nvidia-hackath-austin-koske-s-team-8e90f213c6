from copy import deepcopy
import unittest

from review_pack import blank, validate, select


class ReviewChecks(unittest.TestCase):
    def setUp(self):
        self.m = dict(pack_id='test', samples=[dict(sample_id='a', image_sha256='abc',
            source_run=1, split_role='training_candidate', width=640, height=480)])
        self.doc = blank(self.m, 'manifest')

    def test_blank_has_no_accepted_proposals(self):
        validate(self.doc, self.m, 'manifest')
        self.assertTrue(all(d['status'] == 'unreviewed' and not d['boxes'] for d in self.doc['records'][0]['decisions'].values()))
        bad = deepcopy(self.doc); bad['records'][0]['decisions']['liquid_body']['boxes'] = [self.box()]
        with self.assertRaisesRegex(ValueError, 'Only present'):
            validate(bad, self.m, 'manifest')

    @staticmethod
    def box():
        return dict(xyxy=[300, 350, 330, 400], flags=dict(occluded=False, truncated=False, low_contrast=True))

    def submitted(self):
        d = deepcopy(self.doc); r = d['records'][0]
        r.update(review_status='submitted_for_adjudication', reviewed_by='Aleyna', reviewed_at='2026-10-19T12:00:00Z')
        for name, decision in r['decisions'].items():
            decision['status'] = 'absent'
        r['decisions']['vial'].update(status='present', boxes=[self.box()])
        return d

    def test_review_submission_identity_and_uncertainty_guards(self):
        good = self.submitted(); validate(good, self.m, 'manifest')
        mutations = [lambda d: d.update(training_eligible=True),
            lambda d: d.update(manifest_sha256='other'),
            lambda d: d['records'][0].update(source_run=5),
            lambda d: d['records'][0].update(split_role='development_only'),
            lambda d: d['records'][0].update(reviewed_by=''),
            lambda d: d['records'][0]['decisions']['stream'].update(status='unreviewed'),
            lambda d: d['records'][0]['decisions']['stream'].update(status='uncertain')]
        for mutate in mutations:
            bad = deepcopy(good); mutate(bad)
            with self.assertRaises(ValueError): validate(bad, self.m, 'manifest')
        good['records'][0]['decisions']['stream'].update(status='uncertain', notes='Jet indistinguishable from reflection')
        validate(good, self.m, 'manifest')

    def test_geometry_and_one_station_object(self):
        good = self.submitted()
        for coords in ([330,350,300,400], [300,350,641,400], [300,350,float('nan'),400]):
            bad=deepcopy(good); bad['records'][0]['decisions']['vial']['boxes'][0]['xyxy']=coords
            with self.assertRaises(ValueError): validate(bad,self.m,'manifest')
        bad=deepcopy(good); bad['records'][0]['decisions']['vial']['boxes'].append(self.box())
        with self.assertRaises(ValueError): validate(bad,self.m,'manifest')
        good['records'][0]['decisions']['stream'].update(status='present',boxes=[self.box(),self.box()])
        validate(good,self.m,'manifest')

    def test_deterministic_selection_stays_in_run_and_budget(self):
        rows = [dict(source_id=f'run{r}-{n:03d}', source_run=r, body_height=None if n % 5==0 else n % 25+1,
                     scope_flag=n % 19==0, brightness=n/100, chroma=.1) for r in (1,2,3,4) for n in range(80)]
        a,groups=select(rows);b,again=select(rows)
        self.assertEqual(a,b);self.assertEqual(groups,again);self.assertEqual(len(a),210)
        self.assertEqual([sum(s['source_run']==r for s in a) for r in (1,2,3,4)],[60,60,60,30])
        selected={s['source_id']:s for s in a}
        for g in groups:
            self.assertTrue(all(selected[s]['source_run']==g['source_run'] for s in g['sample_ids']))


if __name__ == '__main__': unittest.main()
