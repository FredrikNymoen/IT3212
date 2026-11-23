"""One readable findings figure and a source-checked thirty-decision ledger."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype']=42
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/assignment4';OUT=BASE/'completion_figures'
ACTIONS=['N','U','R','L','D']
SHORT={
 'TS1/89':'Both score10; the later life loss does not identify the safer route.',
 'TS1/318':'Absent key: zero ties express no learned preference; alternative turn changes without reward.',
 'TS1/634':'Both realise the same movement and gain; requested direction has no unique credit.',
 'TS2/217':'Both gain400; the pickup does not uniquely validate the higher-probability input.',
 'TS2/455':'Both inputs move right; a later life loss cannot identify this action as its cause.',
 'TS2/519':'Near-tied inputs realise the same motion; eventual success does not validate the ranking.',
 'TS3/146':'Near-tied inputs both move upward; apparent preference differs from realised motion.',
 'TS3/393':'Right gains400 versus zero; its later-route risk remains untested.',
 'TS3/401':'Right immediately reaches the target; Left misses immediate success, with eventual outcome unknown.',
 'TF1/188':'NOOP still moves and scores400; Right does too, preventing unique action credit.',
 'TF1/238':'Absent key: zero ties conceal different transitions; their eventual outcomes remain unknown.',
 'TF1/336':'NOOP approaches the ghost; Right moves away. Expected survival is not established.',
 'TF2/217':'Both realise the same movement/gain; the power proxy cannot establish expiry time.',
 'TF2/471':'Near-tied inputs produce the same transition; action probability is not success probability.',
 'TF2/767':'Down approaches the ghost; Up moves away. Later loss alone does not establish expected risk.',
 'TF3/301':'Near-zero estimates miss10 immediately available points; alternative-route success remains untested.',
 'TF3/313':'Both inputs move upward; the later life loss cannot identify which input caused it.',
 'TF3/436':'Right gains100 versus zero: a useful scoring decision inside an unsuccessful episode.',
 'SS1/291':'Absent key: the rare success does not demonstrate reliable value coverage.',
 'SS1/356':'Down moves away; Right approaches. The later loss leaves long-horizon blame unresolved.',
 'SS1/631':'Both gain410; mixed native increments must not be counted as ordinary pellets.',
 'SS2/367':'Key present but five zero values; rare success does not establish an informed preference.',
 'SS2/439':'Down gains400 versus zero: useful local credit despite only9 percent model success.',
 'SS2/523':'Both continue downward near a ghost; later loss does not identify a better full policy.',
 'SF1/326':'Up gains200 versus zero: locally productive action inside a rare failure.',
 'SF1/530':'Near-tied inputs take different paths for equal reward; the long-run ranking remains uncertain.',
 'SF1/625':'Down approaches the ghost; Up moves away. This flags risk without proving failure avoidable.',
 'SF2/205':'Both gain200; high probability establishes neither unique action credit nor eventual success.',
 'SF2/517':'Left moves away, yet a life is later lost; immediate direction cannot settle blame.',
 'SF2/911':'Both leave position unchanged; equal one-step results establish neither inactivity nor successful escape.',
}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pair(v):return '('+','.join(f'{x:.0f}' for x in v)+')'

def build():
    source=BASE/'completion_inspection_v4/inspection.json'
    data=json.loads(source.read_text());episodes={e['case']['id']:e for e in data['episodes']}
    annotations=json.loads((source.parent/'annotations.json').read_text())['annotations']
    comments={a['id']:a for a in annotations};OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    for identity,episode in episodes.items():
        for d in episode['decisions']:
            key=f'{identity}/{d["decision"]}';a=d['alternative'];comment=comments[key]
            records.append(dict(id=key,episode_id=identity,algorithm=episode['case']['job']['algorithm'],
                chosen_action=ACTIONS[d['action']],alternative_action=ACTIONS[a['action']],
                value_labels=[f'{v:.4f}' for v in d['values']],values=d['values'],
                native_frame=d['rgb'],native_frame_sha256=sha(ROOT/d['rgb']),
                interpretation=comment['interpretation'],short_interpretation=re.sub(r'(?<=[A-Za-z])(?=\d)',' ',SHORT[key]),point=d,
                Q_key_present=comment.get('Q_key_present')))
    selected=[('TS1/318','No learned preference'),('TF3/301','Missed immediate points'),
              ('TS3/401','Immediate goal attainment'),('SF1/326','Useful action inside failure')]
    fig=plt.figure(figsize=(7.4,3.9),facecolor='white');shown=[]
    for index,(identity,title) in enumerate(selected):
        record=next(r for r in records if r['id']==identity);d=record['point'];a=d['alternative']
        left=.025+(index%2)*.50;top=.97-(index//2)*.50
        fig.text(left,top,title,fontsize=11,weight='bold',va='top',color='#173b4d')
        fig.text(left,top-.065,identity+' | '+('Q' if record['algorithm']=='Q-learning' else 'PPO'),fontsize=9,va='top')
        ax=fig.add_axes([left,top-.405,.145,.285]);ax.imshow(Image.open(ROOT/d['rgb']))
        x,y=d['player_before'];x0=max(0,min(80,x-40));y0=max(0,min(130,y-40))
        ax.set_xlim(x0,x0+80);ax.set_ylim(y0+80,y0);ax.axis('off')
        symbol='Q' if record['algorithm']=='Q-learning' else 'π'
        for row,(action,gain,position,color,tag) in enumerate([
            (d['action'],d['raw_reward'],d['player_after'],'#176d93','Chosen'),
            (a['action'],a['reward'],a['player'],'#b46c25','Alternative')]):
            ytext=top-.145-row*.145
            fig.text(left+.17,ytext,f'{tag} {ACTIONS[action]}: {symbol} {d["values"][action]:.4f}',fontsize=9.3,color=color,va='top',weight='bold')
            fig.text(left+.17,ytext-.060,f'Gain {gain:g}; move to {pair(position)}',fontsize=8.7,va='top')
        shown.append(dict(id=identity,value_labels=[f'{d["values"][i]:.4f}' for i in [d['action'],a['action']]],
            native_frame=record['native_frame'],native_frame_sha256=record['native_frame_sha256']))
    pdf=OUT/'part4_key_findings.pdf';png=OUT/'part4_key_findings.png'
    fig.savefig(pdf);fig.savefig(png,dpi=200);plt.close(fig)
    figures=[dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=sha(pdf),decisions=shown)]
    assert len(records)==30 and len({r['id'] for r in records})==30
    manifest=dict(source_sha256=sha(source),annotation_sha256=sha(source.parent/'annotations.json'),
        generator_sha256=sha(Path(__file__)),figures=figures,ledger=records,decisions=30,
        legend='Five actions N/U/R/L/D available. Q values scaled reward; PPO values probabilities. Overview has80x80 native crops; full precision and all30 decisions in ledger. Coupled one-step forks do not establish eventual outcomes.')
    (OUT/'episode_figures.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Built one findings figure and a thirty-decision ledger.')
    return manifest

if __name__=='__main__':build()
