"""High-dimensional statistics, static figures, and result-grounded LaTeX."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import re

import numpy as np

from in_depth_analysis import CHECKPOINTS, PROTOCOL, digest, log, read_jsonl, save_json

COLORS = {'robot': '#5689A0', 'ego': '#FF8356'}


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or (norm <= 1e-12).any():
        raise ValueError('Invalid or zero feature vectors')
    return x / norm


def neighbors(x, domain, text, k=20):
    x, text = unit(x), unit(text)
    distances = np.clip(1 - x @ x.T, 0, 2)
    np.fill_diagonal(distances, np.inf)
    # Stable sorting makes tied distances deterministic.
    k = min(k, len(x)-1)
    nn = np.argsort(distances, axis=1, kind='stable')[:, :k]
    mixing = (domain[nn] != domain[:, None]).mean(1)
    cross = distances.copy()
    cross[domain[:, None] == domain[None, :]] = np.inf
    cross_nn = np.argsort(cross, axis=1, kind='stable')[:, :min(k, int(min(np.bincount(domain))))]
    semantic = np.einsum('nd,nkd->nk', text, text[cross_nn]).mean(1)
    return dict(mixing=mixing, semantic=semantic, nearest=cross_nn[:, 0],
                nearest_distance=cross[np.arange(len(x)), cross_nn[:, 0]], k=k)


def feature_health(x):
    from scipy.linalg import svdvals
    x = unit(x)
    centered = x - x.mean(0, keepdims=True)
    singular = svdvals(centered, check_finite=False)
    eig = singular**2
    total = eig.sum()
    if total <= 1e-12:
        return dict(mean_variance=0.0, effective_rank=0.0)
    p = eig[eig > total * 1e-14] / total
    return dict(mean_variance=float(np.var(x, axis=0).mean()),
                effective_rank=float(np.exp(-(p * np.log(p)).sum())))


def interval(values, strata, repeats, seed=42):
    """Stratified episode-score bootstrap, conditional on the fitted kNN graph.

    One sampled frame per original episode is enforced upstream. Resampling
    per-source scores preserves the sampling design and avoids duplicating
    identical points inside kNN searches. This is not retraining uncertainty.
    """
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == key) for key in np.unique(strata)]
    means = []
    for _ in range(repeats):
        index = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        means.append(float(np.mean(values[index])))
    return dict(mean=float(np.mean(values)), ci95=np.quantile(means, [.025, .975]).tolist())


def primitives(text):
    t = text.lower()
    return {
        'translation': bool(re.search(r'\bmove (forward|back|backward|up|down|left|right)\b', t)),
        'rotation': bool(re.search(r'\b(tilt|rotate|turn)\b', t)),
        'hold clause': 'hold position' in t,
        'gripper clause': bool(re.search(r'\b(gripper|grasp|release)\b', t)),
        'bilateral labels': bool(re.search(r'left (wrist|arm|hand|effector)', t) and
                                 re.search(r'right (wrist|arm|hand|effector)', t)),
    }


def savefig(fig, path):
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=220, bbox_inches='tight')


def scatter(ax, xy, domain, title):
    for value, name in [(0, 'robot'), (1, 'ego')]:
        mask = domain == value
        ax.scatter(xy[mask, 0], xy[mask, 1], s=8, c=COLORS[name], alpha=.55,
                   linewidths=0, label=name.capitalize(), rasterized=True)
    ax.set(title=title, xlabel='t-SNE 1', ylabel='t-SNE 2')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, markerscale=2)


def project(x, seed, perplexity):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    x = unit(x)
    x = PCA(n_components=min(50, len(x)-1, x.shape[1]), svd_solver='full').fit_transform(x)
    return TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto',
                max_iter=1500, random_state=seed).fit_transform(x)


def analyze(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from threadpoolctl import threadpool_limits
    # Large shared hosts otherwise oversubscribe BLAS for small matrices.
    threadpool_limits(limits=4)
    out = args.output
    rows = list(read_jsonl(out / 'samples.jsonl'))
    ids = np.array([r['id'] for r in rows])
    if len(set(r['episode_key'] for r in rows)) != len(rows):
        raise ValueError('More than one sampled observation per original episode')
    domain = np.array([int(r['domain'] == 'ego') for r in rows])
    source = np.array([r['source'] for r in rows])
    if domain.sum() != len(domain)//2:
        raise ValueError('Unbalanced domains')
    features, provenance = {}, {}
    for model in ['text', 'without_lap', 'with_lap']:
        metadata = json.loads((out / f'{model}_complete.json').read_text())
        if metadata['identity']['manifest'] != digest(rows):
            raise ValueError(f'Stale manifest in {model}')
        with np.load(metadata['features'], allow_pickle=False) as data:
            if not np.array_equal(data['id'], ids):
                raise ValueError(f'Row alignment mismatch in {model}')
            features[model] = dict(data)
        provenance[model] = metadata
    if not np.array_equal(features['without_lap']['input_hash'], features['with_lap']['input_hash']):
        raise ValueError('Checkpoint comparison used different input tensors')
    x = {'text_raw': features['text']['raw'], 'text_normalized': features['text']['normalized']}
    for model in ['without_lap', 'with_lap']:
        for layer in ['vlm', 'adapter']:
            x[f'{model}_{layer}'] = features[model][layer]
    stats, scores = {}, {}
    for name, array in x.items():
        score = neighbors(array, domain, features['text']['raw'])
        scores[name] = score
        by_group = {}
        for group in ['all', 'robot', 'ego', *np.unique(source)]:
            mask = np.ones(len(rows), dtype=bool) if group == 'all' else (
                domain == int(group == 'ego') if group in ['robot', 'ego'] else source == group)
            by_group[group] = {metric: interval(score[metric][mask], source[mask], args.bootstrap)
                               for metric in ['mixing', 'semantic']}
        stats[name] = dict(groups=by_group, health=feature_health(array), k=score['k'])
    differences = {}
    for layer in ['vlm', 'adapter']:
        differences[layer] = {}
        for metric in ['mixing', 'semantic']:
            delta = scores[f'with_lap_{layer}'][metric] - scores[f'without_lap_{layer}'][metric]
            differences[layer][metric] = interval(delta, source, args.bootstrap)
    primitive_rows = [primitives(r['lap_text']) for r in rows]
    primitive_rates = {group: {key: float(np.mean([p[key] for p, r in zip(primitive_rows, rows) if r['domain'] == group]))
                              for key in primitive_rows[0]} for group in ['robot', 'ego']}
    coverage = {}
    examples = []
    for variant in ['text_raw', 'text_normalized']:
        coverage[variant] = {}
        score = scores[variant]
        for value, name in [(0, 'robot_to_ego'), (1, 'ego_to_robot')]:
            indices = np.flatnonzero(domain == value)
            distances = score['nearest_distance'][indices]
            coverage[variant][name] = dict(zip(['q05', 'q25', 'median', 'q75', 'q95'], np.quantile(distances, [.05, .25, .5, .75, .95]).tolist()))
            order = indices[np.argsort(distances, kind='stable')]
            for category, selected in [('nearest', order[:5]), ('farthest', order[-5:][::-1])]:
                for idx in selected:
                    other = int(score['nearest'][idx])
                    examples.append(dict(variant=variant, direction=name, category=category,
                                         distance=float(score['nearest_distance'][idx]),
                                         query_id=rows[idx]['id'], neighbor_id=rows[other]['id'],
                                         query_text=rows[idx]['lap_text'], neighbor_text=rows[other]['lap_text']))
    result = dict(protocol=PROTOCOL, n=len(rows), manifest_hash=digest(rows),
                  software={name: __import__(name).__version__ for name in ['numpy', 'scipy', 'sklearn', 'matplotlib']},
                  counts=dict(Counter(f'{r["source"]}/{r["split"]}' for r in rows)),
                  metrics=stats, paired_differences=differences, coverage=coverage,
                  primitives=primitive_rates, provenance=provenance,
                  bootstrap=dict(repeats=args.bootstrap, method='source-stratified episode-score bootstrap conditional on fixed neighbor graphs'),
                  caveats=['Checkpoint LAP labels were confirmed by the user; the supplied v0 checkpoint has no training_config.yaml.',
                           'Training steps, data mixtures, action dimensions, and initialization may differ.',
                           'Existing test/val names do not prove training exclusion.',
                           'LAP descriptions use source-specific reference frames and prediction horizons.',
                           'Single-view and stitched camera layouts differ across sources.',
                           'Text overlap and lexical motion coverage do not establish physical action equivalence.'])
    save_json(out / 'metrics.json', result)
    save_json(out / 'nearest_examples.json', examples)
    with (out / 'metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['space', 'group', 'metric', 'mean', 'ci95_low', 'ci95_high'])
        for name, stat in stats.items():
            for group, metrics in stat['groups'].items():
                for metric, val in metrics.items():
                    writer.writerow([name, group, metric, val['mean'], *val['ci95']])
    with (out / 'sample_scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'source', 'space', 'mixing', 'cross_neighbor_text_similarity', 'nearest_distance'])
        for name, score in scores.items():
            for i, row in enumerate(rows):
                writer.writerow([row['id'], row['source'], name, score['mixing'][i], score['semantic'][i], score['nearest_distance'][i]])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    primary = ['text_raw', 'without_lap_vlm', 'with_lap_vlm']
    embeddings = {}
    # Cache each projection by feature bytes and hyperparameters, independent of plots.
    def embedding(name, seed, p):
        key = digest(dict(feature=__import__('hashlib').sha256(x[name].tobytes()).hexdigest(), seed=seed, perplexity=p, iterations=1500))
        path = out / 'projections' / f'{name}_{key}.npy'
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            return np.load(path)
        log(f't-SNE {name}: seed={seed}, perplexity={p}')
        xy = project(x[name], seed, p)
        np.save(path, xy)
        return xy
    p_main = min(30, max(2, (len(rows)-1)//3))
    for name in x:
        embeddings[name] = embedding(name, 42, p_main)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.4))
    scatter(axes[0, 0], embeddings['text_raw'], domain, '(a) LAP motion descriptions')
    keys = list(primitive_rates['robot'])
    for j, name in enumerate(['robot', 'ego']):
        axes[0, 1].barh(np.arange(len(keys)) + (j-.5)*.34,
                       [100*primitive_rates[name][key] for key in keys], height=.34,
                       color=COLORS[name], label=name.capitalize())
    axes[0, 1].set(yticks=np.arange(len(keys)), yticklabels=keys, xlabel='Descriptions containing clause (%)',
                   xlim=(0, 105), title='(b) Lexical motion coverage')
    axes[0, 1].legend(frameon=False)
    scatter(axes[1, 0], embeddings['without_lap_vlm'], domain, '(c) Without LAP (400k)')
    scatter(axes[1, 1], embeddings['with_lap_vlm'], domain, '(d) With LAP (200k)')
    fig.tight_layout()
    savefig(fig, figures / 'in_depth_analysis')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, name, title in zip(axes, ['text_normalized', 'without_lap_adapter', 'with_lap_adapter'],
                               ['Subject-normalized LAP text', 'Without LAP: adapter', 'With LAP: adapter']):
        scatter(ax, embeddings[name], domain, title)
    fig.tight_layout()
    savefig(fig, figures / 'supplementary_features')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, metric, label in zip(axes, ['mixing', 'semantic'], ['Cross-domain neighbor fraction', 'Cross-domain neighbor LAP similarity']):
        for j, model in enumerate(['without_lap', 'with_lap']):
            val = stats[f'{model}_vlm']['groups']['all'][metric]
            ax.errorbar(j, val['mean'], yerr=np.array([[max(0,val['mean']-val['ci95'][0])], [max(0,val['ci95'][1]-val['mean'])]]), fmt='o', capsize=5)
        ax.set(xticks=[0, 1], xticklabels=['Without LAP', 'With LAP'], ylabel=label)
    fig.tight_layout()
    savefig(fig, figures / 'high_dimensional_metrics')
    plt.close(fig)
    sensitivity = []
    for seed in [0, 1, 2]:
        for p in [15, 30, 50]:
            if p >= len(rows):
                continue
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
            for ax, name in zip(axes, primary):
                xy = embedding(name, seed, p)
                scatter(ax, xy, domain, name.replace('_', ' '))
            fig.suptitle(f'Sensitivity: seed {seed}, perplexity {p}')
            fig.tight_layout()
            savefig(fig, figures / f'sensitivity_seed{seed}_p{p}')
            plt.close(fig)
            sensitivity.append(dict(seed=seed, perplexity=p))
    save_json(out / 'analysis_complete.json', dict(n=len(rows), manifest_hash=digest(rows), sensitivity=sensitivity,
              main_projection=dict(seed=42, perplexity=p_main, pca_dims=min(50, len(rows)-1), max_iter=1500)))
    log(f'COMPLETE analysis: {len(rows)} samples')


def report(args):
    out = args.output
    complete = json.loads((out / 'analysis_complete.json').read_text())
    result = json.loads((out / 'metrics.json').read_text())
    if result['manifest_hash'] != complete['manifest_hash']:
        raise ValueError('Stale analysis completion marker')
    n = result['n']//2
    before = result['metrics']['without_lap_vlm']['groups']['all']
    after = result['metrics']['with_lap_vlm']['groups']['all']
    delta = result['paired_differences']['vlm']
    low, high = delta['mixing']['ci95']
    if low > 0:
        outcome = 'The LAP checkpoint exhibits increased cross-domain neighborhood mixing.'
    elif high < 0:
        outcome = 'The LAP checkpoint exhibits reduced cross-domain neighborhood mixing; this comparison does not support increased overlap.'
    else:
        outcome = 'The paired interval includes zero, so this comparison does not establish increased cross-domain neighborhood mixing.'
    if delta['semantic']['ci95'][1] < 0:
        outcome += ' Cross-domain neighbors also have lower LAP-text similarity, which argues against interpreting mixing alone as semantic alignment.'
    else:
        outcome += ' Neighbor-text similarity and feature-rank diagnostics are reported separately; domain mixing alone does not establish semantic alignment.'
    coverage = result['coverage']['text_raw']
    normalized = result['coverage']['text_normalized']
    pr = result['primitives']
    text = rf'''\subsection{{In-depth Analysis}}
\paragraph{{Motion-description coverage.}}
We analyze {n:,} robot observations from AgiBot, DROID, Fractal, and Bridge, and {n:,} human observations from EgoVerse, with at most one observation per original episode.
We encode the associated LAP motion descriptions using a fixed UMT5 encoder, mean-pool valid tokens, and normalize the resulting features.
Figure~\ref{{fig:in_depth_analysis}} visualizes the resulting neighborhoods and the frequency of motion-related clauses.
The median nearest cross-domain cosine distance is {coverage['robot_to_ego']['median']:.3f} from robot to human and {coverage['ego_to_robot']['median']:.3f} in the reverse direction.
After normalizing subject names, the corresponding medians are {normalized['robot_to_ego']['median']:.3f} and {normalized['ego_to_robot']['median']:.3f}.
Despite shared motion vocabulary, the original-space cross-domain neighbor fraction is only {100*result['metrics']['text_raw']['groups']['all']['mixing']['mean']:.2f}\% for raw descriptions and {100*result['metrics']['text_normalized']['groups']['all']['mixing']['mean']:.2f}\% after subject normalization, indicating strong domain separation under this text encoder.
Translation clauses occur in {100*pr['robot']['translation']:.1f}\% of robot descriptions and {100*pr['ego']['translation']:.1f}\% of human descriptions; rotation clauses occur in {100*pr['robot']['rotation']:.1f}\% and {100*pr['ego']['rotation']:.1f}\%, respectively.
Annotation schemas also differ: gripper clauses occur in {100*pr['robot']['gripper clause']:.1f}\% versus {100*pr['ego']['gripper clause']:.1f}\%, while bilateral effector labels occur in {100*pr['robot']['bilateral labels']:.1f}\% versus {100*pr['ego']['bilateral labels']:.1f}\%.
We report deterministic nearest/farthest examples to assess shared motion-language neighborhoods and differences in annotation coverage.
These measurements describe coverage in motion-language space; differing coordinate conventions, action horizons, and annotation styles preclude interpreting them as physical action equivalence.

\paragraph{{Cross-domain representations.}}
We compare the supplied checkpoint without LAP (400k steps) with the checkpoint with LAP (200k steps) on exactly the same observations and task prompts, excluding ground-truth LAP answers.
We extract the final prompt-token representation from the last VLM layer and evaluate cosine neighborhoods before dimensionality reduction.
The fraction of opposite-domain samples among the {result['metrics']['with_lap_vlm']['k']} nearest neighbors changes from {before['mixing']['mean']:.3f} to {after['mixing']['mean']:.3f}, with a paired difference of {delta['mixing']['mean']:+.3f} (95\% interval [{low:+.3f}, {high:+.3f}]).
Mean LAP-text cosine similarity of cross-domain neighbors changes from {before['semantic']['mean']:.3f} to {after['semantic']['mean']:.3f}.
Its paired difference is {delta['semantic']['mean']:+.4f} (95\% interval [{delta['semantic']['ci95'][0]:+.4f}, {delta['semantic']['ci95'][1]:+.4f}]).
{outcome}
Intervals use source-stratified episode-score bootstrap resampling conditional on the fitted neighbor graphs.
Because the checkpoints also differ in training duration, data mixture, and action dimensionality, this is an observational checkpoint comparison rather than an isolated causal ablation of LAP.
Existing split names do not establish that the samples were unseen during training.

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.95\textwidth]{{figures/in_depth_analysis.pdf}}
\caption{{Motion-language coverage and checkpoint representation comparison. Top: frozen UMT5 embeddings of LAP descriptions and lexical motion-clause frequencies. Bottom: last-layer VLM prompt representations without and with LAP. Blue denotes robot observations and orange denotes human observations. Each panel uses a separately fitted t-SNE projection; axes and inter-panel distances are not comparable. Quantitative comparisons use the original feature spaces.}}
\label{{fig:in_depth_analysis}}
\end{{figure*}}
'''
    (out / 'in_depth_analysis.tex').write_text(text)
    summary = ['# In-depth analysis results', '', f'Samples: {result["n"]} ({n} robot, {n} ego).', '',
               '| Metric (VLM) | Without LAP | With LAP | Paired change [95% interval] |',
               '|---|---:|---:|---:|']
    for metric in ['mixing', 'semantic']:
        d = delta[metric]
        summary.append(f'| {metric} | {before[metric]["mean"]:.4f} | {after[metric]["mean"]:.4f} | {d["mean"]:+.4f} [{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}] |')
    summary.extend(['', outcome, '', '## Feature health', '', '| Space | Mean variance | Effective rank |', '|---|---:|---:|'])
    for name, entry in result['metrics'].items():
        h = entry['health']
        summary.append(f'| {name} | {h["mean_variance"]:.6g} | {h["effective_rank"]:.2f} |')
    summary.extend(['', '## Motion-language coverage', '',
                    '| Text variant | Robot to ego median distance | Ego to robot median distance |',
                    '|---|---:|---:|'])
    for variant, values in result['coverage'].items():
        summary.append(f'| {variant} | {values["robot_to_ego"]["median"]:.4f} | {values["ego_to_robot"]["median"]:.4f} |')
    summary.extend(['', f'Original-space cross-domain neighbor fraction: raw text {100*result["metrics"]["text_raw"]["groups"]["all"]["mixing"]["mean"]:.3f}%; subject-normalized text {100*result["metrics"]["text_normalized"]["groups"]["all"]["mixing"]["mean"]:.3f}%.',
                    'Shared translation/rotation vocabulary is present, but this encoder shows strong domain separation. These data do not establish broad motion-language embedding overlap or downstream complementarity benefits.'])
    summary.extend(['', '## Adapter paired changes', ''])
    for metric, value in result['paired_differences']['adapter'].items():
        summary.append(f'- {metric}: {value["mean"]:+.4f}, 95% interval [{value["ci95"][0]:+.4f}, {value["ci95"][1]:+.4f}].')
    summary.extend(['', '## Limits', '', *['- '+c for c in result['caveats']], '',
                    'Intervals condition on fixed neighborhoods and do not include training-run uncertainty.', '',
                    '## Sample splits', '', '```json', json.dumps(result['counts'], indent=2), '```', '',
                    'Main plot parameters and all sensitivity runs are recorded in analysis_complete.json.'])
    (out / 'REPORT.md').write_text('\n'.join(summary) + '\n')
    log('COMPLETE report: REPORT.md and in_depth_analysis.tex')
