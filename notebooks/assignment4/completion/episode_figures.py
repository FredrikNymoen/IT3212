"""Thirty native decision cards: state crops, action values and coupled forks.

Vector labels remain searchable in the PDF. These are scientific figures from
recorded evidence, not screenshots of the old tables. No model is changed.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype']=42
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/assignment4';OUT=BASE/'completion_figures'
PAIRS=[['TS1','TS2'],['TS3','TF1'],['TF2','TF3'],['SS1','SS2'],['SF1','SF2']]
ACTIONS=['N','U','R','L','D']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pair(v):return '('+','.join(f'{x:.0f}' for x in v)+')'

def build():
    source=BASE/'completion_inspection_v4/inspection.json'
    data=json.loads(source.read_text());episodes={e['case']['id']:e for e in data['episodes']}
    annotations=json.loads((source.parent/'annotations.json').read_text())['annotations']
    comments={a['id']:a for a in annotations};OUT.mkdir(parents=True,exist_ok=True)
    records=[];figures=[]
    for page,identities in enumerate(PAIRS,1):
        fig=plt.figure(figsize=(7,6.15),facecolor='white')
        decisions=[]
        for column,identity in enumerate(identities):
            episode=episodes[identity];c=episode['case'];alg=c['job']['algorithm']
            left=.025+column*.50
            outcome='success' if c['success'] else 'failure'
            fig.text(left,.98,f'{identity}  |  {"Q" if alg=="Q-learning" else "PPO"} / {c["job"]["condition"]}  |  {outcome}',
                     fontsize=10,weight='bold',va='top',color='#173b4d')
            fig.text(left,.950,f'{c["score"]:g} points; {c["length"]} decisions; model success {100*c["model_success_rate"]:g}%',fontsize=8.3,va='top')
            for row,d in enumerate(episode['decisions']):
                top=.908-row*.300;identity_step=f'{identity}/{d["decision"]}'
                comment=comments[identity_step];alt=d['alternative'];chosen=d['action'];alternative=alt['action']
                key=(' | Q key '+('present' if comment['Q_key_present'] else 'absent')) if alg=='Q-learning' else ''
                fig.text(left,top,identity_step+key,fontsize=9.2,weight='bold',va='top')
                screen=Image.open(ROOT/d['rgb'])
                ax=fig.add_axes([left,top-.160,.125,.142]);ax.imshow(screen)
                x,y=d['player_before'];x0=max(0,min(96,x-32));y0=max(0,min(146,y-32))
                ax.set_xlim(x0,x0+64);ax.set_ylim(y0+64,y0);ax.axis('off')
                ax=fig.add_axes([left+.174,top-.153,.263,.128])
                values=np.asarray(d['values'],float);lo=min(0,float(values.min()));hi=max(0,float(values.max()))
                span=hi-lo if hi>lo else 1.
                colors=['#b8c2c7']*5;colors[alternative]='#cf792e';colors[chosen]='#176d93'
                ax.barh(range(5),values,color=colors,height=.64)
                ax.set_yticks(range(5),ACTIONS,fontsize=8.5);ax.invert_yaxis()
                ax.set_xlim(lo-.05*span,hi+.45*span);ax.set_xticks([])
                for index,value in enumerate(values):
                    label=f'{value:.4f}'
                    ax.text(max(0,value)+.035*span,index,label,va='center',fontsize=8.3)
                ax.axvline(0,color='#89959a',lw=.5)
                for side in ['top','right','bottom']:ax.spines[side].set_visible(False)
                ax.spines['left'].set_color('#89959a');ax.tick_params(axis='y',length=0,pad=3)
                info=[f'pos {pair(d["player_before"])}; ghost {pair(d["nearest_ghost_offset"])}; food {pair(d["food_offset"])}',
                      f'p={d["encoded_power_proxy"]:.2f}; score {d["score_before"]:g} → {d["score_after"]:g}; lives {d["lives_before"]} → {d["lives_after"]}',
                      f'{ACTIONS[chosen]} → {pair(d["player_after"])}; gain {d["raw_reward"]:g}; task {d["task_reward"]:g}',
                      f'Alt {ACTIONS[alternative]} → {pair(alt["player"])}; gain {alt["reward"]:g}; lives {alt["lives"]}']
                fig.text(left,top-.175,'\n'.join(info),fontsize=8.0,va='top',linespacing=1.25)
                fig.add_artist(plt.Line2D([left,left+.445],[top-.284,top-.284],transform=fig.transFigure,color='#d8e0e3',lw=.7))
                decisions.append(dict(id=identity_step,chosen_action=ACTIONS[chosen],alternative_action=ACTIONS[alternative],
                    value_labels=[f'{v:.4f}' for v in values],values=values.tolist(),state_lines=info,
                    native_frame=d['rgb'],native_frame_sha256=sha(ROOT/d['rgb']),interpretation=comment['interpretation']))
        name=f'part4_episode_pairs_{page:02d}';pdf=OUT/f'{name}.pdf';png=OUT/f'{name}.png'
        fig.savefig(pdf);fig.savefig(png,dpi=190);plt.close(fig)
        figures.append(dict(name=name,episodes=identities,pdf=str(pdf.relative_to(ROOT)),pdf_sha256=sha(pdf),decisions=decisions))
        records.extend(decisions)
    assert len(records)==30 and len({r['id'] for r in records})==30
    manifest=dict(source_sha256=sha(source),annotation_sha256=sha(source.parent/'annotations.json'),
        generator_sha256=sha(Path(__file__)),figures=figures,decisions=30,
        legend='N/U/R/L/D are all available. Blue chosen, orange alternative. Q values are scaled reward; PPO values are probabilities. Native crops64x64; full-frame coordinates; relative ghost/food offsets; p is an imperfect power proxy. Coupled one-step forks do not establish eventual outcomes.')
    (OUT/'episode_figures.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Built five vector figures covering all30 native decisions.')
    return manifest

if __name__=='__main__':build()
