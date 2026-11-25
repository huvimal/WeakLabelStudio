"""Run only this repository's experiment and preserve measured evidence."""
import hashlib
import html
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def source_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}

def main():
    import os
    os.chdir(ROOT)
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    meta=json.loads((ROOT/'project.json').read_text(encoding='utf-8'));start=time.time()
    try:
        from experiment import run
        result=run()
        from common import MODEL_PROVENANCE
        if not isinstance(result,dict) or not result.get('checks') or not result.get('scope'):
            raise ValueError('Experiment must return checks and an explicit scope')
        # JSON compatibility and absence of NaN/Infinity are mandatory.
        json.dumps(result,allow_nan=False)
        versions={}
        for package in ['numpy','Pillow','torch','transformers','onnx','onnxruntime','mcp']:
            try:versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:pass
        report={'project':meta['name'],'number':meta['number'],'status':'local_run_passed',
                'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'duration_seconds':time.time()-start,
                'python':sys.version.split()[0],'platform':platform.platform(),'dependencies':versions,
                'source_sha256':source_hashes(),'result':result,
                'models_used':MODEL_PROVENANCE,
                'publication_note':'Local fixture checks passed, not independent validation or user approval. Review this run before publishing.'}
        (out/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        summary=['# '+meta['name']+' — measured local run','',result['scope'],'','## Results','', '```json',json.dumps(result,indent=2,ensure_ascii=False),'```','',
                 'Local experiment assertions passed. This does not establish production readiness or performance on real customer data. Source fingerprints and dependency versions are in results.json.']
        (out/'RESULTS.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
        page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(meta['name'])+'</title><style>body{font:16px system-ui;background:#0c1524;color:#ecf4ff;max-width:1100px;margin:40px auto;padding:24px}h1{font-size:40px}p{line-height:1.6;color:#bfd0df}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#14283b;border-radius:16px;padding:24px}strong{color:#58e0b8}</style><h1>'+html.escape(meta['name'])+'</h1><p><strong>Measured local experiment</strong> · '+html.escape(report['timestamp_utc'])+'</p><p>'+html.escape(result['scope'])+'</p><pre>'+html.escape(json.dumps(result,indent=2,ensure_ascii=False))+'</pre><p>Authored small fixtures unless explicitly stated otherwise. Inspect errors, assumptions and limitations before sharing. No automatic external actions.</p></html>'
        (out/'report.html').write_text(page,encoding='utf-8')
        (ROOT/'private').mkdir(exist_ok=True)
        (ROOT/'private'/'last-status.json').write_text(json.dumps({'status':'passed'}),encoding='utf-8')
        print(meta['name'].upper()+' CHECKS PASSED')
        print('Scope:',result['scope']);print('Report:',out/'report.html')
    except Exception as exc:
        private=ROOT/'private';private.mkdir(exist_ok=True)
        (private/'last-error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        (private/'last-status.json').write_text(json.dumps({'status':'failed','error':str(exc)}),encoding='utf-8')
        # Previous measured reports remain, but the publish gate rejects a failed latest run.
        print('CHECK STOPPED:',str(exc));return 1
    return 0

if __name__=='__main__':sys.exit(main())
