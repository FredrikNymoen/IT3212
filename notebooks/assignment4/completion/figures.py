"""Scientific figures from measured records/native frames; no generated game imagery."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUTPUT=ROOT/'results/assignment4/completion_figures'


def learning_settings():
    """Equal visual space for A--D; D is a frozen input contrast, not learning."""
    original=ROOT/'results/assignment4/three_hour_20261122/main_v1'
    agents=pd.read_csv(ROOT/'results/assignment4/completion_analysis_v1/agents.csv')
    seeds=[17,29,101,314,2718]
    fig,axes=plt.subplots(2,2,figsize=(10,6.0))
    colors={'Q-learning':'#2476b9','PPO':'#d87426'}
    for ax,condition in zip(axes.ravel()[:3],'ABC'):
        for alg,color in colors.items():
            rows=[pd.read_csv(original/f'{condition}_{alg}_{s}'/'curves.csv') for s in seeds]
            assert all(r.interactions.tolist()==[0,65536,131072,196608,262144] and r.episodes.eq(10).all() for r in rows)
            y=np.array([r.success_rate.to_numpy()*100 for r in rows]);x=rows[0].interactions.to_numpy()
            mean=y.mean(axis=0);sd=y.std(axis=0,ddof=1)
            ax.plot(x,mean,color=color,marker='o',markersize=3,label=alg)
            ax.fill_between(x,np.maximum(0,mean-sd),np.minimum(100,mean+sd),color=color,alpha=.12)
        ax.set_title(f'{condition}: '+{'A':'predictable dynamics','B':'transition uncertainty','C':'step penalty'}[condition],fontsize=11)
        ax.set_xticks([0,131072,262144],['0','131k','262k']);ax.set_xlabel('Training interactions',fontsize=9)
    ax=axes.ravel()[3];d=agents[agents.study.eq('D')]
    for alg,color in colors.items():
        y=[]
        for seed in seeds:
            g=d[d.algorithm.eq(alg)&d.seed.eq(seed)].set_index('condition')
            values=g.loc[['D_clean','D'],'success_rate'].to_numpy()*100;y.append(values)
            ax.plot([0,1],values,color=color,alpha=.25,linewidth=.9,marker='o',markersize=3)
        ax.plot([0,1],np.mean(y,axis=0),color=color,linewidth=2.5,marker='o',markersize=5,label=alg)
    ax.set_xticks([0,1],['Clean','Obstructed']);ax.set_xlim(-.15,1.15)
    ax.set_xlabel('Frozen observation condition (no training)',fontsize=9)
    ax.set_title('D: observation robustness',fontsize=11)
    for ax in axes.ravel():
        ax.set_ylim(0,100);ax.set_ylabel('Success (%)',fontsize=9);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9,loc='upper left');axes[1,1].legend(fontsize=9,loc='upper right')
    fig.tight_layout(pad=1.4);OUTPUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUTPUT/'part2_all_settings.png',dpi=220,bbox_inches='tight');plt.close(fig)
    return OUTPUT/'part2_all_settings.png'


def episode_panels():
    directory=ROOT/'results/assignment4/completion_inspection_v4'
    data=json.loads((directory/'inspection.json').read_text())
    selected=[('TS1',318,'Q: unseen key, five zero values'),('TF3',301,'Q: alternative gains 10, chosen gains 0'),
              ('TS3',401,'PPO: chosen gains 800 and reaches target'),('SF1',625,'PPO: high-probability approach before failure')]
    fig,axes=plt.subplots(1,4,figsize=(10.5,4.0))
    for ax,(identity,step,title) in zip(axes,selected):
        record=next(r for r in data['episodes'] if r['case']['id']==identity)
        point=next(d for d in record['decisions'] if d['decision']==step)
        ax.imshow(Image.open(ROOT/point['rgb']));ax.axis('off')
        ax.set_title(f'{identity}, decision {step}',fontsize=9)
        ax.text(.5,-.04,title.replace(': ',':\n'),transform=ax.transAxes,ha='center',va='top',fontsize=8)
    fig.subplots_adjust(left=.01,right=.99,top=.95,bottom=.17,wspace=.12)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUTPUT/'part4_native_decisions.png',dpi=220,bbox_inches='tight');plt.close(fig)
    return OUTPUT/'part4_native_decisions.png'


def mechanism_curves():
    base=ROOT/'results/assignment4/completion_part3_v1'
    assert (base/'training_completed.json').exists()
    panels=[('Q_fast_epsilon','Q_base','Q: epsilon decay'),('Q_gamma99','Q_base','Q: discount'),
            ('Q_alpha05','Q_base','Q: learning rate'),('PPO_gamma99','PPO_base','PPO: discount'),
            ('PPO_entropy0','PPO_base','PPO: entropy'),('PPO_lr1e4','PPO_base','PPO: learning rate'),
            ('PPO_epochs2','PPO_base','PPO: update epochs')]
    fig,axes=plt.subplots(2,4,figsize=(11.0,5.3),sharex=True,sharey=True)
    for ax,(variant,baseline,title) in zip(axes.ravel(),panels):
        for arm,color,label in [(baseline,'#646464','baseline'),(variant,'#007b84','changed')]:
            values=[]
            for seed in [17,29,101,314,2718]:
                rows=json.loads((base/'training'/f'{arm}_{seed}'/'curves.json').read_text())
                x=np.array([r['interactions'] for r in rows]);values.append([r['success_rate']*100 for r in rows])
            y=np.array(values);mean=y.mean(axis=0);sd=y.std(axis=0,ddof=1)
            ax.plot(x,mean,color=color,marker='o',markersize=2,label=label)
            ax.fill_between(x,np.maximum(0,mean-sd),np.minimum(100,mean+sd),color=color,alpha=.12)
        ax.set_title(title,fontsize=9);ax.grid(alpha=.2);ax.set_ylim(0,100);ax.set_xticks([0,131072,262144],['0','131k','262k'],fontsize=8)
    axes.ravel()[-1].axis('off');axes[0,0].legend(fontsize=8,loc='upper left')
    fig.supxlabel('Real training interactions (Setting B)',fontsize=10);fig.supylabel('Development success (%)',fontsize=10)
    fig.tight_layout(pad=1.1);OUTPUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUTPUT/'part3_mechanism_curves.png',dpi=220,bbox_inches='tight');plt.close(fig)
    return OUTPUT/'part3_mechanism_curves.png'


if __name__=='__main__':
    from episode_figures import build
    build()
    print(learning_settings())
    print(episode_panels())
    if (ROOT/'results/assignment4/completion_part3_v1/training_completed.json').exists():print(mechanism_curves())
