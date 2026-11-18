"""Finalize the user-authorized LDA recommendation and its report exports.

Run after the combined notebook and check_lda_vs_adaboost.py. Original combined
exports are preserved. This follow-up records the revised selection before
scoring LDA on the original held-out students, then regenerates figure assets.
It never compiles report PDFs. The original test had already been inspected,
so this is explicitly an exploratory revision, not a new independent test.
"""
from pathlib import Path
import copy
import json
import shutil
import sys
from time import perf_counter

import check_lda_vs_adaboost as review

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/assignment3/combined'
RESULTS = ROOT / 'results/assignment3/lda_selected'
REVIEW = ROOT / 'results/assignment3/model_choice_review'


def main():
    original_lock, original_environment = review.initialize()
    np, pd = review.np, review.pd
    RESULTS.mkdir(parents=True, exist_ok=True)
    existing_protocol = json.loads((REVIEW / 'protocol.json').read_text())
    assert existing_protocol['selected_LDA'] == 'LDA engineered=True, solver=lsqr, shrinkage=auto'
    candidates = pd.read_csv(REVIEW / 'lda_candidates.csv')
    expected_lda = candidates.sort_values(['mean_CV_AP', 'mean_fold_capacity_recall'], ascending=False).iloc[0]

    # Time the eight-setting LDA search independently of the earlier diagnostic.
    start_search = perf_counter()
    search_rows, pipelines, predictions = [], {}, {}
    for engineered in [False, True]:
        for solver, shrinkage in [('svd', None), ('lsqr', 'auto'), ('lsqr', .1), ('lsqr', .5)]:
            name = f'LDA engineered={engineered}, solver={solver}, shrinkage={shrinkage}'
            pipe = review.model_pipeline(review.LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage),
                engineered=engineered, use_log=review.USE_LOG, rare=review.USE_RARE)
            row, folds, oof = review.evaluate(name, pipe)
            row.update(engineered=engineered, solver=solver, shrinkage=shrinkage)
            search_rows.append(row)
            pipelines[name], predictions[name] = pipe, oof
    trials = pd.DataFrame(search_rows)
    selected = trials.sort_values(['mean_CV_AP', 'mean_fold_capacity_recall'], ascending=False).iloc[0]
    assert selected['model'] == expected_lda['model']
    np.testing.assert_allclose(selected['mean_CV_AP'], expected_lda['mean_CV_AP'], rtol=0, atol=1e-12)
    fitted_lda = review.clone(pipelines[selected['model']])
    refit_start = perf_counter()
    fitted_lda.fit(review.X_dev, review.y_dev)
    lda_refit_seconds = perf_counter() - refit_start
    lda_search_seconds = perf_counter() - start_search
    trials.to_csv(RESULTS / 'search_LDA.csv', index=False)
    lda_oof = predictions[selected['model']]
    alert = review.alert_summary('LDA', lda_oof)
    threshold = alert['OOF_F2_threshold']
    recorded_oof = pd.read_csv(REVIEW / 'development_oof_predictions.csv')
    np.testing.assert_array_equal(review.X_dev.index, recorded_oof.row_id)
    np.testing.assert_allclose(lda_oof, recorded_oof.LDA_score, rtol=0, atol=1e-12)

    # Freeze parameters and development-only policy before reading LDA test scores.
    locked = copy.deepcopy(original_lock)
    locked.update(selected_model='LDA', original_selected_model=original_lock['selected_model'],
        selection_metric='development capacity performance and interpretability trade-off',
        revision='exploratory follow-up after the original test report was inspected',
        test_used_to_select_LDA=False)
    locked['models']['LDA'] = {'parameters': {'engineered': True, 'solver': 'lsqr', 'shrinkage': 'auto'},
                             'threshold': threshold, 'CV_AP': float(selected['mean_CV_AP'])}
    (RESULTS / 'locked_choices.json').write_text(json.dumps(locked, indent=2), encoding='utf-8')

    original_predictions = pd.read_csv(BASE / 'test_predictions.csv')
    test_ids = original_predictions.row_id.to_numpy()
    test_X, test_y = review.X.loc[test_ids], review.y.loc[test_ids]
    np.testing.assert_array_equal(test_y, original_predictions.dropout_label)
    assert len(test_X) == 726 and int(test_y.sum()) == 284
    assert set(test_X.index).isdisjoint(review.X_dev.index)
    lda_score = review.risk_scores(fitted_lda, test_X)
    lda_alert = lda_score >= threshold

    ada = review.model_pipeline(review.AdaBoostClassifier(
        estimator=review.DecisionTreeClassifier(random_state=42), random_state=42),
        engineered=True, use_log=review.USE_LOG, rare=review.USE_RARE)
    ada.set_params(**original_lock['models']['AdaBoost trees']['parameters'])
    start = perf_counter()
    ada.fit(review.X_dev, review.y_dev)
    ada_refit_seconds = perf_counter() - start
    ada_score = review.risk_scores(ada, test_X)
    np.testing.assert_allclose(ada_score, original_predictions['AdaBoost trees_score'], rtol=0, atol=1e-12)

    timing_rows = []
    def benchmark(model, name, expected):
        for _ in range(3):
            np.testing.assert_allclose(review.risk_scores(model, test_X), expected)
        elapsed = []
        for repeat in range(25):
            start = perf_counter()
            repeated = review.risk_scores(model, test_X)
            seconds = perf_counter() - start
            np.testing.assert_allclose(repeated, expected)
            elapsed.append(1000 * seconds / len(test_X))
            timing_rows.append({'model': name, 'repeat': repeat + 1, 'population': 'full_test',
                               'students': len(test_X), 'batch_seconds': seconds,
                               'ms_per_student': elapsed[-1]})
        q25, median, q75 = np.quantile(elapsed, [.25, .5, .75])
        return {'score_ms_per_student': float(median), 'score_ms_q25': float(q25),
                'score_ms_q75': float(q75), 'prediction_repeats': 25}

    lda_timing = benchmark(fitted_lda, 'LDA', lda_score)
    ada_timing = benchmark(ada, 'AdaBoost trees', ada_score)
    pd.DataFrame(timing_rows).to_csv(RESULTS / 'prediction_timing_repeats.csv', index=False)
    coefficients = fitted_lda.named_steps['model'].coef_[0]
    lda_row = {'model': 'LDA', 'AP': review.average_precision_score(test_y, lda_score),
        'ROC_AUC': review.roc_auc_score(test_y, lda_score),
        'accuracy': review.accuracy_score(test_y, lda_alert),
        'balanced_accuracy': review.balanced_accuracy_score(test_y, lda_alert),
        'precision': review.precision_score(test_y, lda_alert), 'recall': review.recall_score(test_y, lda_alert),
        'F1': review.f1_score(test_y, lda_alert), 'F2': review.fbeta_score(test_y, lda_alert, beta=2),
        **review.capacity_metrics(test_y, lda_score), 'search_seconds': lda_search_seconds,
        'refit_seconds': lda_refit_seconds, **lda_timing,
        'complexity': f'{len(coefficients)} linear coefficients; 1 intercept'}
    test_table = pd.read_csv(BASE / 'test_metrics.csv').set_index('model')
    test_table.loc['AdaBoost trees', 'refit_seconds'] = ada_refit_seconds
    for key, value in ada_timing.items():
        test_table.loc['AdaBoost trees', key] = value
    test_table.loc['LDA'] = pd.Series(lda_row).drop('model')
    test_table.to_csv(RESULTS / 'test_metrics.csv')
    original_predictions['LDA_score'], original_predictions['LDA_alert'] = lda_score, lda_alert
    original_predictions.to_csv(RESULTS / 'test_predictions.csv', index=False)
    development = pd.read_csv(BASE / 'development_models.csv').set_index('model')
    development.loc['LDA'] = pd.Series({'family': 'linear classifier', 'CV_AP': selected['mean_CV_AP'],
        'fold_SD': selected['fold_SD'], 'CV_ROC_AUC': float('nan'),
        'CV_recall_capacity': selected['mean_fold_capacity_recall'],
        'search_seconds': lda_search_seconds, 'refit_seconds': lda_refit_seconds,
        'candidates': 8, 'warnings': selected['warnings']})
    development.sort_values('CV_AP', ascending=False).to_csv(RESULTS / 'development_models.csv')
    rules = pd.read_csv(BASE / 'development_oof_rules.csv').set_index('model')
    rules.loc['LDA'] = pd.Series({'OOF_AP': selected['OOF_AP'], 'F2_threshold': threshold,
        'OOF_F2': alert['F2'], **review.capacity_metrics(review.y_dev, lda_oof)})
    rules.to_csv(RESULTS / 'development_oof_rules.csv')
    capacity = pd.DataFrame([{'capacity': fraction, **review.capacity_metrics(test_y, lda_score, fraction)}
                            for fraction in pd.read_csv(BASE / 'capacity_sensitivity.csv').capacity])
    capacity.to_csv(RESULTS / 'capacity_sensitivity.csv', index=False)
    matrix = review.confusion_matrix(test_y, lda_alert)
    summary = {'selected_model': 'LDA', 'test_AP': float(lda_row['AP']),
               'alert_threshold': threshold, 'confusion_matrix': matrix.tolist(),
               'contacts': int(lda_row['contacts']), 'dropouts_reached': int(lda_row['dropouts_reached'])}
    (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    names = fitted_lda.named_steps['preprocessing'].get_feature_names_out()
    names = names[fitted_lda.named_steps['constant'].get_support()]
    pd.DataFrame({'feature': names, 'coefficient': coefficients}).to_csv(RESULTS / 'lda_coefficients.csv', index=False)
    (RESULTS / 'lda_metadata.json').write_text(json.dumps({
        'input_columns': len(coefficients), 'parameters': len(coefficients) + 1,
        'solver': 'lsqr', 'shrinkage': 'auto', 'engineered': True,
        'intercept': float(fitted_lda.named_steps['model'].intercept_[0]),
        'threshold': threshold, 'development_rows': 2904, 'test_rows': 726}, indent=2))
    for name in ['dnn_metadata.json', 'dnn_stopping_history.csv', 'dnn_refit_history.csv',
                 'dnn_seed_sensitivity.csv', 'search_Deep_neural_network.csv']:
        shutil.copy2(BASE / name, RESULTS / name)
    environment = copy.deepcopy(original_environment)
    environment.update(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
        sklearn=review.sklearn.__version__, scipy=review.scipy.__version__,
        torch=review.torch.__version__, timing_provenance={
            'LDA_and_AdaBoost': 'remeasured together in the LDA follow-up',
            'other_models': 'original combined execution',
            'searches': 'original development searches plus added eight-setting LDA search'})
    (RESULTS / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    print('\nFROZEN LDA TEST EVALUATION', flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(test_table.loc[['LDA', 'AdaBoost trees'], ['AP', 'precision', 'recall',
        'contacts', 'dropouts_reached', 'search_seconds', 'refit_seconds', 'score_ms_per_student']].to_string(), flush=True)
    import report_tasks_09_10
    report_tasks_09_10.render(results_dir=RESULTS)
    import report_model_comparison
    report_model_comparison.render(results_dir=RESULTS)


if __name__ == '__main__':
    main()
