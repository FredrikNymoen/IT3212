const byId = id => document.getElementById(id);
const ui = Object.fromEntries([
  'experiment', 'refresh', 'intro', 'download', 'notice', 'dashboard', 'settings', 'setting-detail',
  'stats', 'model', 'episode', 'outcome', 'poster', 'video', 'play', 'screen-message', 'replay-data',
  'replay-error', 'metric', 'curve', 'chart-empty', 'chart-legend', 'chart-detail', 'agent-rows', 'run-details',
].map(id => [id, byId(id)]));

const settingNames = { A: 'Forutsigbar', B: 'Usikker', C: 'Stegstraff', D: 'Transfer' };
const outcomes = { success: 'Suksess', game_over: 'Spillet slutt', timeout: 'Tidsgrense', maze_exit: 'Labyrintbytte' };
const actionNames = { NOOP: 'Ingen', UP: 'Opp', RIGHT: 'Høyre', LEFT: 'Venstre', DOWN: 'Ned',
  UPRIGHT: 'Opp–høyre', UPLEFT: 'Opp–venstre', DOWNRIGHT: 'Ned–høyre', DOWNLEFT: 'Ned–venstre' };
const colors = { DQN: '#59dac5', PPO: '#ba91fb', 'Q-table': '#ffad71',
  'Q-table-pellet': '#ffd958', 'PPO-low-lr': '#7eb7ff' };
const color = (algorithm, index = 2) => colors[algorithm] || ['#7eb7ff', '#f7a472', '#ffd958'][index % 3];
const state = { agents: [], episodes: [], data: null, setting: null, agent: null, episode: null,
  replay: null, hasTrace: false, experimentSeq: 0, episodeSeq: 0, replaySeq: 0, chartPoint: -1, chartScale: null };
const originalQuery = new URLSearchParams(location.search);

const present = value => value !== null && value !== undefined && Number.isFinite(Number(value));
const num = (value, digits = 0) => present(value)
  ? Number(value).toLocaleString('nb-NO', { maximumFractionDigits: digits }) : '–';
const percent = value => present(value) ? `${num(value * 100, 2)} %` : '–';
const seconds = value => present(value) ? `${num(value, 1)} s` : '–';
const label = agent => `${agent.algorithm} · frø ${agent.train_seed}`;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function option(value, text) {
  const node = element('option', text);
  node.value = value;
  return node;
}

async function json(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || 'Kunne ikke hente data.');
  return payload;
}

function setNotice(message) {
  ui.notice.textContent = message;
  ui.notice.hidden = !message;
}

function showScreenMessage(message) {
  ui['screen-message'].textContent = message;
  ui['screen-message'].hidden = !message;
}

function replayError(message) {
  ui['replay-error'].textContent = message;
  ui['replay-error'].hidden = !message;
}

function dateLabel(id) {
  const parts = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})\d*Z$/.exec(id);
  if (!parts) return id;
  const [, year, month, date, hour, min, sec] = parts;
  return new Intl.DateTimeFormat('nb-NO', { dateStyle: 'medium', timeStyle: 'medium' })
    .format(new Date(`${year}-${month}-${date}T${hour}:${min}:${sec}Z`));
}

function updateUrl() {
  const url = new URL(location.href);
  for (const [key, value] of Object.entries({ run: state.experiment, setting: state.setting,
    agent: state.agent?.eval_id, seed: state.episode?.eval_seed })) {
    if (value === undefined || value === null) url.searchParams.delete(key);
    else url.searchParams.set(key, value);
  }
  history.replaceState(null, '', url);
}

const base = () => `/api/experiments/${encodeURIComponent(state.experiment)}`;
function episodeUrl(operation) {
  const query = new URLSearchParams({ agent: state.agent.eval_id });
  if (operation !== 'episodes') query.set('seed', state.episode.eval_seed);
  return `${base()}/${operation}?${query}`;
}

async function loadList(preferred, restore = {}) {
  ui.refresh.disabled = true;
  ui.experiment.disabled = true;
  try {
    const experiments = await json('/api/experiments');
    ui.experiment.replaceChildren(...experiments.map(run => option(run.id,
      [dateLabel(run.id), run.label, run.ready ? null : 'pågår'].filter(Boolean).join(' · '))));
    ui.experiment.disabled = !experiments.length;
    if (!experiments.length) {
      ui.dashboard.hidden = true;
      ui.download.hidden = true;
      setNotice('Ingen kjøringer funnet i part2_results. Kjør notebooken og oppdater listen.');
      return;
    }
    const selected = experiments.find(run => run.id === preferred) || experiments.find(run => run.ready) || experiments[0];
    ui.experiment.value = selected.id;
    await loadExperiment(selected.id, restore);
  } catch (error) {
    setNotice(error.message);
  } finally {
    ui.refresh.disabled = false;
  }
}

async function loadExperiment(id, restore = {}) {
  const sequence = ++state.experimentSeq;
  ++state.episodeSeq;
  resetPlayer();
  state.experiment = id;
  state.data = state.setting = state.agent = state.episode = null;
  ui.dashboard.hidden = true;
  ui.download.hidden = true;
  setNotice('Laster kjøringen …');
  try {
    const data = await json(base());
    if (sequence !== state.experimentSeq) return;
    state.data = data;
    if (!data.agents.length) {
      setNotice('Kjøringen har ingen ferdige sluttevalueringer ennå. Oppdater listen når notebooken er ferdig.');
      updateUrl();
      return;
    }
    const settings = [...new Set([...Object.keys(data.manifest.settings || {}), ...data.agents.map(agent => agent.setting)])];
    const scoreBand = ui.metric.querySelector('option[value="score_1000_rate"]');
    scoreBand.disabled = !data.curves.some(row => present(row.score_1000_rate));
    if (scoreBand.disabled && ui.metric.value === scoreBand.value) ui.metric.value = 'mean_raw_score';
    ui.settings.replaceChildren(...settings.map(setting => {
      const tab = element('button', undefined, 'tab');
      tab.type = 'button';
      tab.dataset.setting = setting;
      tab.append(element('span', setting, 'tab-key'), document.createTextNode(settingNames[setting] || 'Innstilling'));
      tab.addEventListener('click', () => selectSetting(setting));
      return tab;
    }));
    const algorithms = [...new Set(data.agents.map(agent => agent.algorithm))];
    const names = new Intl.ListFormat('nb-NO').format(algorithms);
    ui.intro.textContent = `${names} i ${data.manifest.env_id} · mål på ${num(data.manifest.success_points)} spillpoeng.`;
    ui.download.href = `${base()}/summary.csv`;
    ui.download.hidden = !data.summary.length;
    setNotice('');
    ui.dashboard.hidden = false;
    const initial = settings.includes(restore.setting) ? restore.setting :
      data.agents.some(agent => agent.setting === 'B') ? 'B' : data.agents[0].setting;
    await selectSetting(initial, restore);
  } catch (error) {
    if (sequence === state.experimentSeq) setNotice(error.message);
  }
}

function selectSetting(setting, restore = {}) {
  state.setting = setting;
  state.agents = state.data.agents.filter(agent => agent.setting === setting);
  state.chartPoint = -1;
  for (const tab of ui.settings.children) tab.setAttribute('aria-pressed', String(tab.dataset.setting === setting));
  const cfg = state.data.manifest.settings?.[setting];
  const detail = cfg ? [`${percent(cfg.sticky)} gjentakelsessannsynlighet`, `labyrint ${Number(cfg.maze) + 1}`] : [];
  if (cfg?.step_penalty) detail.push(`${num(cfg.step_penalty, 5)} i trekk per steg`);
  if (cfg?.score_bonus) detail.push(`${num(cfg.score_bonus, 3)} i bonus per ${num(state.data.manifest.success_points)} spillpoeng`);
  if (setting === 'D') detail.push('gjenbruker B-modellene');
  ui['setting-detail'].textContent = detail.join(' · ');
  renderStats();
  drawChart();
  ui.model.replaceChildren(...state.agents.map(agent => option(agent.eval_id, label(agent))));
  ui.model.disabled = !state.agents.length;
  const best = [...state.agents].sort((a, b) => b.success_rate - a.success_rate || b.mean_raw_score - a.mean_raw_score)[0];
  const requested = state.agents.find(agent => agent.eval_id === restore.agent);
  return selectModel(requested || best, restore.seed);
}

function renderStats() {
  const agents = state.agents;
  const budgets = [...new Set(agents.map(agent => agent.training_interactions))].sort((a, b) => a - b);
  const successful = [...agents].sort((a, b) => b.success_rate - a.success_rate)[0];
  const highest = [...agents].sort((a, b) => b.mean_raw_score - a.mean_raw_score)[0];
  const trials = agents.reduce((sum, agent) => sum + agent.episodes, 0);
  const agentCount = `${num(agents.length)} evaluerte modeller`;
  const stats = [
    ['Trening per modell', budgets.length > 1 ? `${num(budgets[0])}–${num(budgets.at(-1))}` : num(budgets[0]),
      state.setting === 'D' ? 'Interaksjoner brukt i B' : 'Interaksjoner med miljøet'],
    ['Testepisoder', num(trials), agentCount],
    ['Beste suksessrate', percent(successful?.success_rate), successful?.success_rate ? label(successful) : 'Ingen suksesser'],
    ['Høyeste snittscore', num(highest?.mean_raw_score, 1), highest ? label(highest) : 'Ingen modeller'],
  ];
  ui.stats.replaceChildren(...stats.map(([title, value, caption], i) => {
    const stat = element('article', undefined, i === 2 ? 'stat stat-highlight' : 'stat');
    stat.append(element('h2', title, 'stat-label'), element('p', value, 'stat-value'), element('p', caption, 'stat-caption'));
    return stat;
  }));
}

function renderTable() {
  const rows = state.agents.map(agent => {
    const tr = element('tr');
    if (state.agent === agent) tr.className = 'selected';
    const td = element('td');
    const button = element('button', undefined, 'model-link');
    button.type = 'button';
    button.setAttribute('aria-label', `Se episoder fra ${label(agent)}`);
    button.setAttribute('aria-pressed', String(state.agent === agent));
    const dot = element('i', undefined, 'model-dot');
    dot.style.setProperty('--series', color(agent.algorithm));
    button.append(dot, document.createTextNode(agent.algorithm));
    button.addEventListener('click', () => {
      selectModel(agent);
      ui.model.focus({ preventScroll: true });
      ui.model.closest('.panel').scrollIntoView({ block: 'nearest' });
    });
    td.append(button);
    const values = [num(agent.train_seed), percent(agent.score_1000_rate), percent(agent.success_rate), percent(agent.failure_rate), num(agent.mean_raw_score, 1),
      num(agent.mean_return, 4), num(agent.success_length, 1), num(agent.failure_length, 1), seconds(agent.training_seconds)];
    tr.append(td, ...values.map(value => element('td', value)));
    return tr;
  });
  if (!rows.length) {
    const tr = element('tr');
    const td = element('td', 'Ingen modeller evaluert i denne innstillingen.');
    td.colSpan = 10;
    tr.append(td);
    rows.push(tr);
  }
  ui['agent-rows'].replaceChildren(...rows);
}

function definitionList(items) {
  const list = element('dl');
  for (const [name, value] of items) {
    list.append(element('dt', name), element('dd', value));
  }
  return list;
}

function renderDetails() {
  const { manifest, runs, summary } = state.data;
  const agent = state.agent;
  const cfg = agent && runs[agent.run_id];
  const hardware = manifest.hardware || {};
  const settings = [
    ['Kjørings-ID', state.experiment], ['Miljø', manifest.env_id], ['Suksess ved', `${num(manifest.success_points)} poeng`],
    ['Tidsgrense', `${num(manifest.max_episode_steps)} interaksjoner`],
    ['Treningsfrø', manifest.training_seeds?.join(', ') || '–'],
    ['Maskinvare', [hardware.cpu, hardware.device].filter(Boolean).join(' · ') || '–'],
    ...Object.entries(manifest.software || {}).filter(([name]) => ['gymnasium', 'ale-py', 'torch'].includes(name)),
  ];
  const model = agent ? [
    ['Modell-ID', agent.run_id], ['Evalueringsregel', agent.policy || '–'],
    ['Evalueringstid', seconds(agent.eval_seconds)], ['Fullførte treningsepisoder', num(agent.completed_training_episodes)],
    ['Optimizer-oppdateringer', num(agent.optimizer_updates)],
    ...(present(agent.q_table_states) ? [['Tilstander i Q-tabellen', num(agent.q_table_states)]] : []),
    ...(present(agent.known_state_rate) ? [['Kjente tilstander ved evaluering', percent(agent.known_state_rate)]] : []),
    ...Object.entries(cfg?.params || {}).map(([key, value]) => [key, typeof value === 'number' ? num(value, 8) : String(value)]),
  ] : [];
  const algorithms = summary.filter(row => row.setting === state.setting).flatMap(row => [
    [`${row.algorithm} · suksess`, `${percent(row.success_rate)}${present(row.success_rate_sd) ? ` ± ${num(row.success_rate_sd * 100, 2)} prosentpoeng` : ''}`],
    [`${row.algorithm} · spillpoeng`, `${num(row.mean_raw_score, 1)} ± ${num(row.mean_raw_score_sd, 1)}`],
  ]);
  ui['run-details'].replaceChildren(...[
    ['Eksperiment', settings], ['Valgt modell', model], ['Gjennomsnitt ± standardavvik mellom frø', algorithms],
  ].filter(([, items]) => items.length).map(([heading, items]) => {
    const block = element('section');
    block.append(element('h3', heading), definitionList(items));
    return block;
  }));
}

async function selectModel(agent, preferredSeed) {
  const sequence = ++state.episodeSeq;
  state.agent = agent;
  state.episode = null;
  state.episodes = [];
  state.hasTrace = false;
  ui.model.value = agent?.eval_id ?? '';
  ui.episode.replaceChildren(option('', agent ? 'Laster episoder …' : 'Ingen episoder'));
  ui.episode.disabled = true;
  resetPlayer();
  renderTable();
  renderDetails();
  updateUrl();
  if (!agent) {
    showScreenMessage('Ingen modell å spille av i denne innstillingen.');
    return;
  }
  showScreenMessage('Laster episoder …');
  try {
    const { episodes, has_trace: hasTrace } = await json(episodeUrl('episodes'));
    if (sequence !== state.episodeSeq) return;
    state.episodes = episodes;
    state.hasTrace = hasTrace;
    const ordered = [...episodes].sort((a, b) =>
      Number(b.outcome === 'success') - Number(a.outcome === 'success') || a.eval_seed - b.eval_seed);
    ui.episode.replaceChildren(...ordered.map(episode => option(String(episode.eval_seed),
      `${episode.eval_seed} · ${num(episode.raw_score)} p · ${outcomes[episode.outcome] || episode.outcome}`)));
    ui.episode.disabled = !ordered.length;
    selectEpisode(ordered.find(episode => String(episode.eval_seed) === String(preferredSeed)) || ordered[0]);
  } catch (error) {
    if (sequence !== state.episodeSeq) return;
    ui.episode.replaceChildren(option('', 'Kunne ikke hente episoder'));
    showScreenMessage(error.message);
  }
}

function resetPlayer() {
  ++state.replaySeq;
  state.replay = null;
  ui.video.pause();
  ui.video.hidden = true;
  if (ui.video.hasAttribute('src')) {
    ui.video.removeAttribute('src');
    ui.video.load();
  }
  ui.poster.hidden = true;
  ui.poster.removeAttribute('src');
  ui.play.hidden = true;
  ui.play.disabled = false;
  ui.outcome.hidden = true;
  ui['replay-data'].replaceChildren();
  replayError('');
  showScreenMessage('');
}

function selectEpisode(episode) {
  resetPlayer();
  state.episode = episode;
  updateUrl();
  if (!episode) {
    showScreenMessage('Denne modellen har ingen lagrede episoder.');
    return;
  }
  ui.episode.value = String(episode.eval_seed);
  ui.outcome.textContent = outcomes[episode.outcome] || episode.outcome;
  ui.outcome.classList.toggle('success', episode.outcome === 'success');
  ui.outcome.hidden = false;
  renderReplayData();
  if (state.hasTrace) {
    ui.poster.src = episodeUrl('poster');
    ui.poster.hidden = false;
    ui.play.hidden = false;
  } else {
    showScreenMessage('Denne kjøringen mangler lagrede handlingsspor.');
  }
}

function renderReplayData() {
  if (!state.episode) return;
  const timeline = state.replay?.timeline;
  const step = timeline ? Math.min(timeline.length - 1, Math.floor(ui.video.currentTime * state.replay.fps + 0.001)) : 0;
  const frame = timeline?.[step];
  const items = [
    [frame ? 'Spillpoeng' : 'Sluttpoeng', num(frame?.score ?? state.episode.raw_score)],
    ['Steg', frame ? `${num(step)} / ${num(state.episode.length)}` : num(state.episode.length)],
    ['Liv', num(frame?.lives)],
    ['Handling', actionNames[frame?.action] || frame?.action || '–'],
  ];
  if (!ui['replay-data'].children.length) {
    ui['replay-data'].replaceChildren(...items.map(([key, value]) => {
      const cell = element('div');
      cell.append(element('dt', key), element('dd', value));
      return cell;
    }));
  } else {
    items.forEach(([key, value], i) => {
      const [dt, dd] = ui['replay-data'].children[i].children;
      dt.textContent = key;
      dd.textContent = value;
    });
  }
}

async function playEpisode() {
  const sequence = ++state.replaySeq;
  ui.play.disabled = true;
  showScreenMessage('Gjenskaper episoden i emulatoren …');
  replayError('');
  try {
    const replay = await json(episodeUrl('replay'));
    if (sequence !== state.replaySeq) return;
    state.replay = replay;
    ui.video.src = replay.video;
    ui.video.hidden = false;
    ui.poster.hidden = true;
    ui.play.hidden = true;
    showScreenMessage('');
    renderReplayData();
    try {
      await ui.video.play();
    } catch (error) {
      if (sequence === state.replaySeq && error.name !== 'AbortError') {
        replayError('Trykk på avspillingsknappen i videoen for å starte.');
      }
    }
  } catch (error) {
    if (sequence !== state.replaySeq) return;
    showScreenMessage('');
    replayError(error.message);
    ui.play.disabled = false;
  }
}

function groupBy(items, key) {
  const groups = new Map();
  for (const item of items) {
    const value = key(item);
    if (!groups.has(value)) groups.set(value, []);
    groups.get(value).push(item);
  }
  return groups;
}

function curveSeries(metric) {
  const ids = new Set(state.agents.map(agent => agent.run_id));
  const groups = groupBy(state.data.curves.filter(row => ids.has(row.run_id) && present(row[metric])), row => row.algorithm);
  return [...groups].map(([algorithm, rows], i) => ({ algorithm, color: color(algorithm, i), points:
    [...groupBy(rows, row => row.interactions)].sort(([a], [b]) => a - b).map(([x, group]) => {
      const values = group.map(row => row[metric]);
      const y = values.reduce((sum, next) => sum + next, 0) / values.length;
      const sd = values.length > 1 ? Math.sqrt(values.reduce((sum, value) => sum + (value - y) ** 2, 0) / (values.length - 1)) : 0;
      return { x, y, sd, n: values.length };
    }),
  }));
}

function tickInterval(span) {
  const exponent = 10 ** Math.floor(Math.log10(span / 4));
  return ([1, 2, 2.5, 5, 10].find(n => n >= span / (4 * exponent)) || 10) * exponent;
}

function drawChart() {
  if (!state.data || ui.dashboard.hidden) return;
  const metric = ui.metric.value;
  const series = curveSeries(metric);
  const canvas = ui.curve;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  ui['chart-empty'].hidden = Boolean(series.length);
  if (!series.length) {
    ui['chart-empty'].textContent = 'Ingen læringskurve lagret for denne innstillingen.';
    ui['chart-legend'].replaceChildren();
    ui['chart-detail'].textContent = '';
    state.chartScale = null;
    return;
  }
  const points = series.flatMap(line => line.points);
  const steps = [...new Set(points.map(point => point.x))].sort((a, b) => a - b);
  const maxX = Math.max(steps.at(-1), 1);
  const rate = metric === 'success_rate' || metric === 'score_1000_rate';
  const upperBound = rate ? 1 : Infinity;
  const lowerBound = metric === 'mean_return' ? -Infinity : 0;
  const min = Math.max(lowerBound, Math.min(0, ...points.map(point => point.y - point.sd)));
  const max = Math.min(upperBound, Math.max(rate ? .01 : 0, ...points.map(point => point.y + point.sd)));
  const interval = tickInterval(max - min || 1);
  const bottom = Math.floor(min / interval) * interval;
  const top = Math.min(upperBound, Math.max(bottom + interval, Math.ceil(max / interval) * interval));
  const pad = { left: 56, right: 17, top: 14, bottom: 44 };
  const x = value => pad.left + value / maxX * (width - pad.left - pad.right);
  const y = value => pad.top + (top - value) / (top - bottom) * (height - pad.top - pad.bottom);
  state.chartScale = { steps, x };
  const plotValue = value => rate ? percent(value) : num(value, metric === 'mean_return' ? 3 : 0);

  ctx.strokeStyle = '#33343e';
  ctx.fillStyle = '#b1aebc';
  ctx.font = '12px "Aptos", "Segoe UI", sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let value = bottom; value <= top + interval * 0.01; value += interval) {
    const pos = y(value);
    ctx.beginPath();
    ctx.moveTo(pad.left, pos);
    ctx.lineTo(width - pad.right, pos);
    ctx.stroke();
    ctx.fillText(plotValue(value), pad.left - 9, pos);
  }
  ctx.textAlign = 'center';
  const labelEvery = Math.max(1, Math.ceil(steps.length / (width < 370 ? 4 : 6)));
  steps.forEach((value, index) => {
    if (index % labelEvery === 0) ctx.fillText(num(value), x(value), height - pad.bottom + 15);
  });
  ctx.fillText('Treningsinteraksjoner', (pad.left + width - pad.right) / 2, height - 8);

  for (const line of series) {
    const section = boundary => boundary.forEach(([point, value], index) =>
      index === 0 ? ctx.moveTo(x(point.x), y(value)) : ctx.lineTo(x(point.x), y(value)));
    const band = [
      ...line.points.map(point => [point, Math.min(top, point.y + point.sd)]),
      ...[...line.points].reverse().map(point => [point, Math.max(bottom, point.y - point.sd)]),
    ];
    ctx.beginPath();
    section(band);
    ctx.closePath();
    ctx.fillStyle = `${line.color}27`;
    ctx.fill();
    ctx.beginPath();
    section(line.points.map(point => [point, point.y]));
    ctx.strokeStyle = line.color;
    ctx.lineWidth = 2.2;
    ctx.stroke();
    ctx.fillStyle = line.color;
    for (const point of line.points) {
      ctx.beginPath();
      ctx.arc(x(point.x), y(point.y), points.length < 28 ? 3.2 : 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  const selectedStep = steps[state.chartPoint];
  const source = state.setting === 'D' ? 'B-agentenes treningskurver. ' : '';
  ui['chart-detail'].textContent = `${source}Gjennomsnitt per algoritme. Skyggen viser ett standardavvik mellom treningsfrøene.`;
  if (selectedStep !== undefined) {
    const cross = x(selectedStep);
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#a6a5b1';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cross, pad.top);
    ctx.lineTo(cross, height - pad.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    const labels = series.flatMap(line => line.points.filter(point => point.x === selectedStep)
      .map(point => `${line.algorithm} ${plotValue(point.y)} (${point.n} frø)`));
    ui['chart-detail'].textContent = `${num(selectedStep)} interaksjoner · ${labels.join(' · ')}`;
  }
  ui['chart-legend'].replaceChildren(...series.map(line => {
    const item = element('span');
    const dot = element('i');
    dot.style.setProperty('--series', line.color);
    item.append(dot, document.createTextNode(line.algorithm));
    return item;
  }));
}

ui.experiment.addEventListener('change', () => loadExperiment(ui.experiment.value));
ui.refresh.addEventListener('click', () => loadList(state.experiment, {
  setting: state.setting, agent: state.agent?.eval_id, seed: state.episode?.eval_seed,
}));
ui.model.addEventListener('change', () => selectModel(state.agents.find(agent => agent.eval_id === ui.model.value)));
ui.episode.addEventListener('change', () => selectEpisode(state.episodes.find(episode => String(episode.eval_seed) === ui.episode.value)));
ui.play.addEventListener('click', playEpisode);
ui.poster.addEventListener('error', () => {
  ui.poster.hidden = true;
  if (state.hasTrace && state.episode) replayError('Forhåndsbildet kunne ikke lastes. Du kan fortsatt prøve å spille av episoden.');
});
for (const event of ['timeupdate', 'seeked', 'ended']) ui.video.addEventListener(event, renderReplayData);
ui.video.addEventListener('playing', () => replayError(''));
ui.video.addEventListener('error', () => {
  if (!state.replay) return;
  replayError('Nettleseren kunne ikke spille av videoen. Prøv å velge episoden på nytt.');
});
ui.metric.addEventListener('change', () => { state.chartPoint = -1; drawChart(); });
ui.curve.tabIndex = 0;
ui.curve.addEventListener('pointermove', event => {
  const scale = state.chartScale;
  if (!scale) return;
  const position = event.clientX - ui.curve.getBoundingClientRect().left;
  const index = scale.steps.reduce((best, step, i) =>
    Math.abs(position - scale.x(step)) < Math.abs(position - scale.x(scale.steps[best])) ? i : best, 0);
  if (state.chartPoint !== index) { state.chartPoint = index; drawChart(); }
});
ui.curve.addEventListener('pointerleave', () => { state.chartPoint = -1; drawChart(); });
ui.curve.addEventListener('keydown', event => {
  if (!state.chartScale || !['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
  event.preventDefault();
  const next = state.chartPoint + (event.key === 'ArrowRight' ? 1 : -1);
  state.chartPoint = Math.min(state.chartScale.steps.length - 1, Math.max(0, next));
  drawChart();
});
new ResizeObserver(drawChart).observe(ui.curve.parentElement);

loadList(originalQuery.get('run'), Object.fromEntries(originalQuery));
