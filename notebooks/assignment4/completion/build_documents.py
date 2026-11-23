"""Compile separate submission documents and render every page for visual QA."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import re
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[3]
def main():
    report=ROOT/'reports/assignment4';build=ROOT/'results/assignment4/completion_latex_build'
    output=ROOT/'output/pdf';build.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    previews=ROOT/'results/assignment4/completion_pdf_preview';previews.mkdir(exist_ok=True)
    # Build beside private auxiliary files. A stale main.aux in the editable
    # report directory otherwise takes precedence over -output-directory.
    for path in report.rglob('*'):
        if path.is_file() and (path.suffix in {'.tex','.png'} or path.parent==report/'Images' and path.suffix=='.pdf'):
            dest=build/path.relative_to(report);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    documents={'main':'report','evidence_brief':'evidence_brief','predictions':'predictions','review':'review_status','ai_record':'ai_record'}
    result=[]
    for source,target in documents.items():
        for iteration in range(2):
            command=['pdflatex','-interaction=nonstopmode','-halt-on-error',source+'.tex']
            completed=subprocess.run(command,cwd=build,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace')
            (build/f'{source}_pass{iteration}.txt').write_text(completed.stdout,encoding='utf-8')
            assert completed.returncode==0,completed.stdout[-3000:]
        log=(build/f'{source}.log').read_text(errors='replace')
        assert not any(s in log for s in ['Overfull \\hbox','undefined on input line','There were undefined references','Missing character:','Fatal error','Infinite glue shrinkage']),source
        dest=output/f'assignment4_{target}.pdf';shutil.copy2(build/f'{source}.pdf',dest)
        reader=PdfReader(dest);texts=[page.extract_text() for page in reader.pages]
        assert all(text.strip() for text in texts) and all('??' not in text for text in texts)
        visible=re.sub(r'\s+',' ',' '.join(texts))
        if source=='main':
            assert visible.count('Part 5: Human review')==1
            assert visible.index('Inspect what aggregate scores hide')<visible.index('Part 5: Human review')<visible.index('Make three recommendations')
        if source=='evidence_brief':assert 'Part 5: Human review' not in visible
        subprocess.run(['pdftoppm','-r','75','-png',str(dest),str(previews/target)],check=True,stdout=subprocess.DEVNULL)
        result.append(dict(document=source,pdf=str(dest.relative_to(ROOT)),pages=len(reader.pages),
                           sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),latex_errors=False))
    (build/'compiled.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
