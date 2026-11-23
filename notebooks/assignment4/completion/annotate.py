"""Individual Part 4 commentary tied to verified recorded decisions, not scripts of imagined play."""
import json
from pathlib import Path
import pickle
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parent/'pilot')]
import q_learning as q

COMMENTS={
 'TS1/89': 'Left approaches the nearby ghost with zero power proxy while collecting 10 points. Right also collects 10; the life counter falls 21 decisions later. Immediate reward does not settle route risk.',
 'TS1/318': 'All five Q-values tie at zero. Seeded tie-breaking selects Up; Right changes the immediate path without a reward difference. This is not a confident learned preference.',
 'TS1/634': 'Up narrowly outranks Right, yet both inputs realise the same rightward position and 400-point gain. Requested direction is not necessarily executed direction.',
 'TS2/217': 'Right has probability .402 versus Up .313. Both inputs yield 400 in this prefix; that pickup does not uniquely validate the higher-probability input.',
 'TS2/455': 'Right receives .609 with zero power proxy and a ghost to the right. The Up alternative also moves right. A life decrement 21 decisions later cannot be attributed to this input alone.',
 'TS2/519': 'Left and Down probabilities differ by less than .001. Both inputs realise the same leftward position; the final success does not establish that this tiny ranking matters.',
 'TS3/146': 'Left and Up are almost tied, and both realise upward movement. The apparent turn preference and the observed motion must be distinguished.',
 'TS3/393': 'Right produces 400 while Down produces zero and a different position. Right is locally productive in this coupled fork; later-route risk is untested.',
 'TS3/401': 'Right produces 800 and crosses the target; Left produces zero. The alternative would miss immediate success, but its eventual outcome is unknown.',
 'TF1/188': 'NOOP has the highest Q-value and still moves right, collecting 400. Right also collects 400. NOOP does not mean that MsPacMan stops.',
 'TF1/238': 'All Q-values tie at zero. Right realises downward movement, whereas Up moves upward. Equal estimates conceal genuinely different transitions.',
 'TF1/336': 'NOOP continues left toward the nearby ghost with zero power proxy; Right moves away with equal immediate reward. This is an avoidable local approach, not proof of lower expected survival.',
 'TF2/217': 'Down has probability .417; the power proxy is .13, not a verified expiry time. Down and Up both realise the same leftward position and 200-point gain.',
 'TF2/471': 'NOOP .334 narrowly beats Down .327. Both realise the same motion with zero reward; action probability is not success probability.',
 'TF2/767': 'Down receives .757 and moves toward the nearby ghost; Up moves away. The later life loss is real, but one coupled step cannot establish the safer expected route.',
 'TF3/301': 'Near-zero tied estimates choose Down for zero points; Left produces 10. This verifies a missed immediate opportunity, not a full alternative success.',
 'TF3/313': 'Right has the highest Q-value but realises upward movement, as does Up. The later life decrement does not prove that selecting Right caused it.',
 'TF3/436': 'Right produces 100 versus zero for Down despite a small Q-value gap. A useful local scoring decision still belongs to an unsuccessful episode.',
 'SS1/291': 'A zero-valued tie selects NOOP. Up changes the immediate turn without gaining points. One rare successful trajectory does not establish reliable value coverage.',
 'SS1/356': 'Down only slightly outranks Up and moves away from the nearest ghost; Right instead moves upward toward it. The actual later life loss leaves long-horizon blame unresolved.',
 'SS1/631': 'NOOP is strongly preferred, continues left and produces 410, as does Left. Mixed native score increments must not be interpreted as counts of ordinary pellets.',
 'SS2/367': 'A five-way zero tie selects Up; Right produces a different turn with equal reward. The eventual rare success does not validate an informed preference here.',
 'SS2/439': 'Down is the sole positive Q estimate and produces 400, while Up produces zero. This local credit is useful although the model succeeds in only 9 percent of episodes.',
 'SS2/523': 'NOOP continues downward near a ghost with zero power proxy, as does Down. Both yield zero immediately; the later life loss cannot identify a better full policy.',
 'SF1/326': 'Up produces 200 whereas Right produces zero. This locally productive decision occurs in a rare failure; failed episodes need not consist of uniformly poor actions.',
 'SF1/530': 'Up .337 and Right .336 are almost tied but produce different paths. Both yield zero, leaving the long-run ranking uncertain.',
 'SF1/625': 'Down receives .836 and moves toward the ghost; Up moves away with equal reward. This is a candidate risky choice, not proof that the failure was inevitable or avoidable.',
 'SF2/205': 'Left receives .874 and produces 200; Down realises the same position and gain. The large probability does not establish unique action credit or eventual success.',
 'SF2/517': 'Left and Up are almost tied. Left moves away from the ghost, yet the life counter falls later. Immediate direction alone cannot separate policy error from later events.',
 'SF2/911': 'Down and Up have almost equal probabilities and both leave position unchanged. Their equal one-step results do not establish intentional inactivity or a successful escape.',
}


def main():
    directory=q.ROOT/'results/assignment4/completion_inspection_v4'
    source=directory/'inspection.json';data=json.loads(source.read_text());assert data['source_sha256']==q.sha(HERE/'inspect_episodes.py')
    output=[]
    for row in data['episodes']:
        case=row['case'];job=case['job'];model=None;observations=None
        if job['algorithm']=='Q-learning':
            path=q.ROOT/job['model'];assert q.sha(path)==job['model_sha256']
            with path.open('rb') as handle:model=pickle.load(handle)
            with np.load(Path(job['directory'])/'evaluation_traces.npz') as archive:
                observations=archive['observation'][archive['episode']==case['episode_index']]
        for decision in row['decisions']:
            identity=f'{case["id"]}/{decision["decision"]}';assert identity in COMMENTS
            record=dict(id=identity,case_id=case['id'],decision=decision['decision'],interpretation=COMMENTS[identity])
            if model is not None:
                state=model.state(observations[decision['decision']-1])
                assert np.array_equal(model.values(state),np.array(decision['values'],dtype=np.float32))
                record.update(Q_key=list(map(int,state[1])),Q_key_present=state[1] in model.table)
            output.append(record)
    assert len(output)==30 and set(COMMENTS)=={r['id'] for r in output}
    q.write_json(directory/'annotations.json',dict(annotations=output,inspection_sha256=q.sha(source),annotation_source_sha256=q.sha(__file__),
        scope='Retrospective episode interpretation, not human peer review. Alternative outcomes are conditional one-step replays, not expected returns.'))
    print(json.dumps(dict(annotations=30,zero_tie_states=[r for r in output if 'Q_key_present' in r and not r['Q_key_present']]),indent=2))


if __name__=='__main__':main()
