"""Read-only inventory of actual preparation/review evidence; no label approval."""
import argparse,csv,json,sqlite3
from collections import Counter
from pathlib import Path
from liquid_specialists_data import write,sha


def main(a):
 if a.out.exists():raise ValueError('Use a fresh output')
 a.out.mkdir(parents=True)
 standards=a.repo/'datasets/mixing_twin/results/physical_experiment_plan_20261010/independent_color_standards.csv'
 trial=a.repo/'datasets/mixing_twin/results/physical_experiment_plan_20261010/physical_trial_worksheet.csv'
 with standards.open(newline='',encoding='utf-8-sig') as f:sr=list(csv.DictReader(f))
 with trial.open(newline='',encoding='utf-8-sig') as f:tr=list(csv.DictReader(f))
 # These counts assess documentation completeness, not whether a liquid is acceptable.
 preparations=[r for r in sr if r.get('preparation_method') and r.get('actual_total_ml') and r.get('camera_file')]
 measured=[r for r in tr if r.get('composition_measurement_method') and r.get('camera_file') and r.get('timestamp')]
 databases=[]
 for folder in (a.tools/'results/annotation-review-20261019',a.tools/'results/aleyna-review-editor-qa'):
  if not folder.exists():continue
  for path in folder.rglob('*.sqlite'):
   item=dict(path=str(path),access='sqlite_mode_ro')
   try:
    with sqlite3.connect('file:'+path.resolve().as_posix()+'?mode=ro',uri=True) as db:
     tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
     counts=Counter();eligible=0
     if 'revisions' in tables:
      for (payload,) in db.execute('SELECT payload FROM revisions'):
       record=json.loads(payload);counts.update([record.get('reviewer_kind','unknown')]);eligible+=int(record.get('eligible_as_reviewed_label') is True)
     item.update(revision_kinds=dict(counts),eligible_spatial_revision_records=eligible)
   except (sqlite3.Error,ValueError) as error:item['read_error']=type(error).__name__
   databases.append(item)
 supplied=sorted(p.name for p in (a.repo/'datasets/dataset2/csi-vial-fill-dataset/vial_fill/labels').glob('*'))
 report=dict(schema_version=1,kind='evidence_inventory_not_human_adjudication',
     source_files={str(standards):sha(standards),str(trial):sha(trial)},
     standard_worksheet_rows=len(sr),rows_with_recorded_preparation_and_camera= len(preparations),
     physical_trial_worksheet_rows=len(tr),rows_with_recorded_composition_method_camera_and_time=len(measured),
     supplied_label_filenames=supplied,run5_values_loaded=False,review_database_inventory=databases,
     independently_approved_recipe_reference_bank_found=False,
     limits=['Scoped inventory of supplied files and existing local review workspaces, not every user directory.',
         'Spatial revision eligibility is not product acceptance, liquid-mask truth or recipe certification.',
         'Blank worksheet cells are not experimental results; no historical recipe/quality label was inferred.'])
 write(a.out/'completed.json',report)
 print(json.dumps({k:v for k,v in report.items() if k not in ('review_database_inventory','source_files')},indent=2))
 print('Read-only review databases:',len(databases))


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('repo','tools','out'):p.add_argument('--'+n,type=Path,required=True)
 main(p.parse_args())
