"""Lock selected variants and train a balanced, fresh-seed complete A/B/C cohort."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import pandas as pd
import cohort

study=cohort.study

def main(args):
    selected_path=study.ROOT/'results/assignment4'/args.selection_lock
    selected=json.loads(selected_path.read_text())
    assert selected['stage']=='lock'
    variants={algorithm:selected['algorithms'][algorithm][0]['variant'] for algorithm in ['Q-learning','PPO']}
    seeds=[int(x) for x in args.seeds.split(',')]
    assert len(set(seeds))==len(seeds) and not set(seeds)&{42,7},'Use fresh main training seeds.'
    assert args.budget%65536==0
    output=study.ROOT/'results/assignment4'/args.run_id;output.mkdir(parents=True,exist_ok=True)
    hashes=study.sources();hashes[str(Path(__file__).relative_to(study.ROOT))]=study.sha(__file__)
    jobs=[];points=list(range(0,args.budget+1,65536))
    # PPO has the longer measured jobs. Scheduling these first reduces the tail;
    # every algorithm/setting still has identical budgets and seed lists.
    for algorithm in ['PPO','Q-learning']:
        for setting in 'ABC':
            for seed in seeds:
                jobs.append(dict(algorithm=algorithm,variant=variants[algorithm],setting=setting,seed=seed,budget=args.budget,
                    checkpoints=points,development_seeds=list(range(1130000,1130010)),
                    directory=str(output/f'{setting}_{algorithm}_{seed}'),source_hashes=hashes))
    protocol=dict(jobs=jobs,source_hashes=hashes,workers=args.workers,training_seeds=seeds,budget=args.budget,
        selected_variants=variants,selection_lock=str(selected_path.relative_to(study.ROOT)),selection_lock_sha256=study.sha(selected_path),
        budget_reason='Common rollout-compatible cap based on late source-pilot improvement/regression and measured throughput within the approved three hours; not a convergence claim.',
        main_stopping_rule='Same source-condition development success/score/earlier-checkpoint rule per trained model; every run consumes the full cap; B checkpoint reused on D.',
        scope='Five fresh main seeds if runtime permits; earlier inspected outcomes/unequal historical exploration disclosed. No D training or selection.',
        reward='Unchanged native score increments A/B, .1-point decision penalty only in C; internal learning rewards /100.')
    path=output/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Preserve changed-source/configuration runs.'
    else:
        study.write(path,protocol)
        for name in hashes:
            target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((study.ROOT/name).read_bytes())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(study.worker,job) for job in jobs]):
            result=future.result()
            print('MAIN DONE '+json.dumps({key:result[key] for key in ('algorithm','variant','setting','seed','training_seconds')}),flush=True)
    comparison=pd.concat([pd.read_csv(Path(job['directory'])/'curves.csv').assign(algorithm=job['algorithm'],
        variant=job['variant'],setting=job['setting'],seed=job['seed']) for job in jobs],ignore_index=True)
    comparison.to_csv(output/'comparison.csv',index=False)
    study.write(output/'completed.json',dict(jobs=len(jobs),training_interactions=sum(job['budget'] for job in jobs),
        source_hashes_verified=True,selection_lock_sha256=study.sha(selected_path)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selection-lock',required=True);parser.add_argument('--run-id',required=True)
    parser.add_argument('--budget',type=int,default=393216);parser.add_argument('--seeds',default='17,29,101,314,2718')
    parser.add_argument('--workers',type=int,default=8);main(parser.parse_args())
