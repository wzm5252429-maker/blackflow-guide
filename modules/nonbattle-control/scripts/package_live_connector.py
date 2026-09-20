"""Freeze the live connector's explicit dependency surface, without training runs."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(output: Path):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stage = output / 'blackflow-live-connector'
    if stage.exists():
        raise ValueError('Use a new output directory; an existing frozen package is never overwritten')
    stage.mkdir()
    files = list((ROOT/'blackflow_live').glob('*.py'))
    files += list((ROOT/'blackflow_live/assets').glob('*'))
    # Keep independent research work out of releases. New inference dependencies
    # require an explicit change to this reviewed list, not a folder-wide glob.
    dependency_list = ROOT/'scripts/live_inference_files.json'
    dependencies = json.loads(dependency_list.read_text(encoding='utf-8'))['files']
    if len(dependencies) != len(set(dependencies)):
        raise ValueError('Duplicate inference dependency')
    for relative in dependencies:
        path = (ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'Missing or invalid inference dependency: {relative}')
        files.append(path)
    files.append(dependency_list)
    selection_path = ROOT/'data/policies/current_neural_controller.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    files += [selection_path]
    for field in ('checkpoint','menu_checkpoint','profile'):
        path = (ROOT/selection[field]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'Missing or invalid selected {field}')
        if sha256(path.read_bytes()).hexdigest() != selection[field+'_sha256']:
            raise ValueError(f'Selected {field} does not match its pinned hash')
        files.append(path)
    files += list((ROOT/'tests').glob('test_live_*.py'))
    for fixture in ('live_hud', 'live_nodes', 'live_rewards', 'live_shop_dialog', 'live_recruitment', 'live_recruitment_cards'):
        files += [p for p in (ROOT/'tests/fixtures'/fixture).glob('*') if p.suffix in {'.png','.jpg','.json'}]
    files += [ROOT/p for p in ('requirements-core.txt','requirements-live.txt','tools/Start-BlackflowLive.cmd',
        'docs/nonbattle-control.md','docs/live-policy-coverage.md','scripts/package_live_connector.py')]
    vision_docs = ROOT/'docs/live-vision-coverage.md'
    if vision_docs.is_file():
        files.append(vision_docs)
    entries = []
    for path in sorted(set(files)):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        dest = stage/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        entries.append({'path':relative.as_posix(),'sha256':sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
    (stage/'tests/__init__.py').write_text('',encoding='utf-8')
    (stage/'.gitattributes').write_text('* -text\n',encoding='utf-8')
    shutil.copyfile(ROOT/'docs/nonbattle-control.md',stage/'README.md')
    manifest = {'schema_version':1,'policy':selection['name'],'files':entries,
                'scope':'first ending nonbattle; no claimed complete real-game acceptance',
                'maa_assets_included':False,'source':'local frozen inference dependency snapshot'}
    (stage/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive = output/'blackflow-live-connector.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                z.write(path,path.relative_to(stage).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError('Archive validation failed')
    print(json.dumps({'stage':str(stage),'archive':str(archive),'files':len(entries),'bytes':archive.stat().st_size},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    build(parser.parse_args().output)
