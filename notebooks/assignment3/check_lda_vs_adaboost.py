"""Development-only check of LDA classification versus the locked AdaBoost.

Run from any directory with the Assignment 3 dependencies installed. Reuses
the combined notebook's data audit, exact splits, feature engineering and
preprocessor definitions, without executing its searches or test evaluation.
Outputs stay separate under results/assignment3/model_choice_review/.
No notebook, report, image, PDF or original result file is overwritten.
"""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]

import json
from time import perf_counter
import warnings

import numpy as np
import pandas as pd

COMPARISON_RESULTS = ROOT / 'results/assignment3/model_choice_review'
COMPARISON_RESULTS.mkdir(parents=True, exist_ok=True)
NOTEBOOK = ROOT / 'notebooks/assignment3/student_dropout_combined.ipynb'
ORIGINAL_RESULTS = ROOT / 'results/assignment3/combined'


def notebook_source(marker):
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    matches = [''.join(cell['source']) for cell in notebook['cells']
               if cell['cell_type'] == 'code' and marker in ''.join(cell['source'])]
    assert len(matches) == 1, marker
    return matches[0]


def initialize():
    os.chdir(NOTEBOOK.parent)
    setup = notebook_source('def profile_holdout(').split('\ncpu_name =')[0]
    imports, configuration = setup.split('SEED = 42', 1)
    exec(compile(imports, str(NOTEBOOK) + ':imports', 'exec'), globals())
    globals()['display'] = lambda *args, **kwargs: None
    configuration = ('SEED = 42' + configuration).replace(
        "RESULTS = Path('../../results/assignment3/combined')",
        'RESULTS = COMPARISON_RESULTS').replace(
        'FIGURES.mkdir(parents=True, exist_ok=True)',
        '# This check does not create or update figures.')
    assert 'RESULTS = COMPARISON_RESULTS' in configuration
    exec(compile(configuration, str(NOTEBOOK) + ':setup', 'exec'), globals())
    definitions = notebook_source('class AcademicFeatures(').split(
        '\nprobe = LogisticRegression')[0]
    exec(compile(definitions, str(NOTEBOOK) + ':preprocessing', 'exec'), globals())

    locked = json.loads((ORIGINAL_RESULTS / 'locked_choices.json').read_text())
    globals()['USE_LOG'] = locked['log']
    globals()['USE_RARE'] = locked['rare_pooling']
    recorded = pd.read_csv(ORIGINAL_RESULTS / 'partition_membership.csv')
    assert set(recorded.loc[recorded.partition.eq('development'), 'row_id']) == set(X_dev.index)
    assert len(X_dev) == 2904 and int(y_dev.sum()) == 1137
    original_environment = json.loads((ORIGINAL_RESULTS / 'environment.json').read_text())
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == original_environment['data_sha256']
    # Prevent accidental use of the held-out population in this diagnostic.
    for key in ['X_test', 'y_test', 'test_indices']:
        globals().pop(key)
    return locked, original_environment


def evaluate(name, pipeline):
    scores = np.full(len(X_dev), np.nan)
    rows = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for fold, (train, valid) in enumerate(FOLDS):
            fitted = clone(pipeline)
            start = perf_counter()
            fitted.fit(X_dev.iloc[train], y_dev.iloc[train])
            seconds = perf_counter() - start
            predicted = risk_scores(fitted, X_dev.iloc[valid])
            scores[valid] = predicted
            rows.append({'model': name, 'fold': fold,
                         'AP': average_precision_score(y_dev.iloc[valid], predicted),
                         'ROC_AUC': roc_auc_score(y_dev.iloc[valid], predicted),
                         'fit_seconds': seconds,
                         **capacity_metrics(y_dev.iloc[valid], predicted)})
    assert np.isfinite(scores).all()
    frame = pd.DataFrame(rows)
    summary = {'model': name, 'mean_CV_AP': float(frame.AP.mean()),
               'fold_SD': float(frame.AP.std(ddof=0)),
               'mean_fold_capacity_recall': float(frame.recall_at_capacity.mean()),
               'OOF_AP': float(average_precision_score(y_dev, scores)),
               'mean_fold_fit_seconds': float(frame.fit_seconds.mean()),
               'warnings': len(caught),
               **capacity_metrics(y_dev, scores)}
    print(f'{name}: CV AP={summary["mean_CV_AP"]:.5f}; '
          f'OOF contacts={summary["dropouts_reached"]}/{summary["contacts"]}', flush=True)
    return summary, frame, scores


def measure_final_fit(name, pipeline):
    fitted = clone(pipeline)
    start = perf_counter()
    fitted.fit(X_dev, y_dev)
    fit_seconds = perf_counter() - start
    expected = risk_scores(fitted, X_dev)
    for _ in range(3):
        np.testing.assert_allclose(risk_scores(fitted, X_dev), expected)
    timing = []
    for repeat in range(25):
        start = perf_counter()
        score = risk_scores(fitted, X_dev)
        seconds = perf_counter() - start
        np.testing.assert_allclose(score, expected)
        timing.append({'model': name, 'repeat': repeat + 1,
                       'students': len(X_dev), 'batch_seconds': seconds,
                       'ms_per_student': 1000 * seconds / len(X_dev)})
    frame = pd.DataFrame(timing)
    return fitted, frame, {'model': name, 'fit_seconds': fit_seconds,
                           'prediction_ms_median': float(frame.ms_per_student.median()),
                           'prediction_ms_q25': float(frame.ms_per_student.quantile(.25)),
                           'prediction_ms_q75': float(frame.ms_per_student.quantile(.75))}


def alert_summary(name, score):
    precision, recall, thresholds = precision_recall_curve(y_dev, score)
    f2 = np.divide(5 * precision[:-1] * recall[:-1],
                   4 * precision[:-1] + recall[:-1], out=np.zeros(len(thresholds)),
                   where=(4 * precision[:-1] + recall[:-1]) > 0)
    threshold = float(thresholds[np.argmax(f2)])
    alert = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y_dev, alert).ravel()
    return {'model': name, 'OOF_F2_threshold': threshold,
            'alerts': int(alert.sum()), 'precision': float(precision_score(y_dev, alert)),
            'recall': float(recall_score(y_dev, alert)),
            'F2': float(fbeta_score(y_dev, alert, beta=2)),
            'true_negative': int(tn), 'false_alert': int(fp),
            'missed_dropout': int(fn), 'true_positive': int(tp)}


def paired_bootstrap(lda_scores, ada_scores):
    # Identical profiles are resampled together, including the conflicting-label pair.
    groups = pd.util.hash_pandas_object(X_dev, index=False).to_numpy()
    unique = np.unique(groups)
    positions = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(42)
    rows = []
    for repeat in range(1000):
        indices = np.concatenate([positions[i] for i in rng.integers(len(unique), size=len(unique))])
        truth = y_dev.to_numpy()[indices]
        lda, ada = lda_scores[indices], ada_scores[indices]
        rows.append({'repeat': repeat + 1,
                     'AP_difference_LDA_minus_AdaBoost': average_precision_score(truth, lda)
                     - average_precision_score(truth, ada),
                     'capacity_recall_difference_LDA_minus_AdaBoost':
                     capacity_metrics(truth, lda)['recall_at_capacity']
                     - capacity_metrics(truth, ada)['recall_at_capacity']})
    frame = pd.DataFrame(rows)
    frame.to_csv(COMPARISON_RESULTS / 'conditional_paired_bootstrap.csv', index=False)
    return {column: {'lower_95': float(frame[column].quantile(.025)),
                     'upper_95': float(frame[column].quantile(.975))}
            for column in frame.columns if column != 'repeat'}


def main():
    locked, original_environment = initialize()
    started = perf_counter()
    lda_options = [('svd', None), ('lsqr', 'auto'), ('lsqr', .1), ('lsqr', .5)]
    candidates, details, oof_scores, pipelines = [], [], {}, {}
    for engineered in [False, True]:
        for solver, shrinkage in lda_options:
            name = f'LDA engineered={engineered}, solver={solver}, shrinkage={shrinkage}'
            pipeline = model_pipeline(
                LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage),
                engineered=engineered, use_log=USE_LOG, rare=USE_RARE)
            summary, folds, score = evaluate(name, pipeline)
            summary.update(engineered=engineered, solver=solver, shrinkage=shrinkage)
            candidates.append(summary)
            details.append(folds)
            oof_scores[name], pipelines[name] = score, pipeline

    candidate_table = pd.DataFrame(candidates)
    candidate_table.to_csv(COMPARISON_RESULTS / 'lda_candidates.csv', index=False)
    # Same candidate selection rule as the combined workflow; no test scores enter it.
    selected = candidate_table.sort_values(
        ['mean_CV_AP', 'mean_fold_capacity_recall'], ascending=False).iloc[0]
    selected_name = selected['model']

    ada = model_pipeline(AdaBoostClassifier(
        estimator=DecisionTreeClassifier(random_state=42), random_state=42),
        engineered=True, use_log=USE_LOG, rare=USE_RARE)
    ada.set_params(**locked['models']['AdaBoost trees']['parameters'])
    ada_summary, ada_folds, ada_oof = evaluate('AdaBoost locked configuration', ada)
    expected_ap = locked['models']['AdaBoost trees']['CV_AP']
    np.testing.assert_allclose(ada_summary['mean_CV_AP'], expected_ap, atol=1e-12, rtol=0)
    expected_trials = pd.read_csv(ORIGINAL_RESULTS / 'search_AdaBoost_trees.csv')
    expected_row = expected_trials.iloc[np.argmin(np.abs(expected_trials.mean_test_AP - expected_ap))]
    np.testing.assert_allclose(ada_folds.AP,
        [expected_row[f'split{i}_test_AP'] for i in range(5)], atol=1e-12, rtol=0)
    details.append(ada_folds)
    pd.concat(details).to_csv(COMPARISON_RESULTS / 'fold_results.csv', index=False)
    comparison = pd.DataFrame([selected.to_dict(), ada_summary])
    comparison.to_csv(COMPARISON_RESULTS / 'comparison.csv', index=False)
    predictions = pd.DataFrame({'row_id': X_dev.index, 'dropout_label': y_dev.to_numpy(),
                                'LDA_score': oof_scores[selected_name], 'AdaBoost_score': ada_oof})
    predictions.to_csv(COMPARISON_RESULTS / 'development_oof_predictions.csv', index=False)
    alert_results = pd.DataFrame([
        alert_summary('LDA selected configuration', oof_scores[selected_name]),
        alert_summary('AdaBoost locked configuration', ada_oof)])
    alert_results.to_csv(COMPARISON_RESULTS / 'oof_alert_comparison.csv', index=False)
    capacity_count = int(np.ceil(.2 * len(X_dev)))
    lda_contacts = set(np.argsort(-oof_scores[selected_name], kind='stable')[:capacity_count])
    ada_contacts = set(np.argsort(-ada_oof, kind='stable')[:capacity_count])
    overlap = {'contacts_per_model': capacity_count,
               'same_students_contacted': len(lda_contacts & ada_contacts),
               'students_unique_to_each_list': len(lda_contacts - ada_contacts)}
    (COMPARISON_RESULTS / 'contact_overlap.json').write_text(json.dumps(overlap, indent=2))

    uncertainty = paired_bootstrap(oof_scores[selected_name], ada_oof)
    costs, raw_timing = [], []
    fitted_lda = None
    for name, pipeline in [('LDA selected configuration', pipelines[selected_name]),
                           ('AdaBoost locked configuration', ada)]:
        fitted, timing, cost = measure_final_fit(name, pipeline)
        costs.append(cost)
        raw_timing.append(timing)
        if name.startswith('LDA'):
            fitted_lda = fitted
    pd.DataFrame(costs).to_csv(COMPARISON_RESULTS / 'timing_comparison.csv', index=False)
    pd.concat(raw_timing).to_csv(COMPARISON_RESULTS / 'prediction_timing_repeats.csv', index=False)
    feature_names = fitted_lda.named_steps['preprocessing'].get_feature_names_out()
    feature_names = feature_names[fitted_lda.named_steps['constant'].get_support()]
    coefficients = fitted_lda.named_steps['model'].coef_[0]
    assert len(feature_names) == len(coefficients)
    pd.DataFrame({'transformed_feature': feature_names, 'Dropout_log_odds_coefficient': coefficients}).to_csv(
        COMPARISON_RESULTS / 'lda_coefficients.csv', index=False)
    linear_score = fitted_lda[:-1].transform(X_dev) @ coefficients + fitted_lda.named_steps['model'].intercept_[0]
    np.testing.assert_allclose(linear_score, fitted_lda.decision_function(X_dev))

    protocol = {'purpose': 'exploratory model-choice check after the original test results were known',
                'selection_population': 'development only', 'held_out_test_evaluated': False,
                'development_rows': len(X_dev), 'development_dropouts': int(y_dev.sum()),
                'folds': 5, 'split_seed': 42, 'same_original_group_folds': True,
                'shared_preprocessing': {'log': USE_LOG, 'rare_pooling': USE_RARE,
                                         'fitted_within_training_folds': True},
                'LDA_candidates': 8, 'LDA_selection': 'mean fold AP, then capacity recall',
                'selected_LDA': selected_name,
                'AdaBoost': 'original locked configuration; original fold AP reproduced exactly',
                'bootstrap': {'repeats': 1000, 'seed': 42,
                              'method': 'paired group percentile bootstrap of fixed OOF predictions',
                              'limitations': 'conditional on fitted predictions; selected CV reuse; no retraining uncertainty',
                              'intervals': uncertainty},
                'timing': 'fresh full-development fits; 3 warm-ups and 25 batch calls on development features; no resubstitution accuracy used',
                'alerts': 'F2 threshold optimized and described on the same OOF predictions; optimistic development diagnostic',
                'environment': {'python': sys.version, 'numpy': np.__version__,
                                'pandas': pd.__version__, 'sklearn': sklearn.__version__,
                                'cpu': platform.processor(), 'platform': platform.platform(),
                                'original_run_cpu': original_environment['cpu'], 'numerical_thread_limit': 2},
                'LDA_coefficients': len(coefficients),
                'LDA_intercept': float(fitted_lda.named_steps['model'].intercept_[0]),
                'total_seconds': perf_counter() - started}
    (COMPARISON_RESULTS / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print('\nSELECTED COMPARISON', flush=True)
    print(comparison[['model', 'mean_CV_AP', 'mean_fold_capacity_recall',
                      'OOF_AP', 'contacts', 'dropouts_reached', 'precision_at_capacity']].to_string(index=False), flush=True)
    print('\nFRESH TIMING COMPARISON', flush=True)
    print(pd.DataFrame(costs).to_string(index=False), flush=True)
    print('\nOOF ALERT DIAGNOSTIC (thresholds selected on these same OOF predictions)', flush=True)
    print(alert_results.to_string(index=False), flush=True)
    print('\nCONTACT OVERLAP', json.dumps(overlap), flush=True)
    print('\nCONDITIONAL PAIRED INTERVALS', json.dumps(uncertainty, indent=2), flush=True)
    print(f'Finished in {protocol["total_seconds"]:.1f}s. Outputs: {COMPARISON_RESULTS}', flush=True)


if __name__ == '__main__':
    main()
