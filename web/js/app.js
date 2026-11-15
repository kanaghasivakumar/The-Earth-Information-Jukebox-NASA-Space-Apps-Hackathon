// Earth Information Jukebox: UI glue. Plain ES modules, no build step.
import { Engine } from './audio.js';
import { Announcer, speak, cancelSpeech, canSpeak, sleep } from './speech.js';
import { Roll } from './roll.js';
import {
  GROUPS, GROUP_LABELS, PitchMap, fmtTime, spokenTime, fmtValue, spokenValue, noteName, fmtNum, spokenNum,
} from './music.js';

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => {
  const e = Object.assign(document.createElement(tag), props);
  for (const k of kids) e.append(k);
  return e;
};
const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

const engine = new Engine();
const announcer = new Announcer($('live'), $('live-status'));
const roll = new Roll($('roll'), { onSeek: (t) => seek(t, true) });
const video = $('video');
const still = $('still');

const S = {
  index: [], track: null, base: '', timeline: [], controls: null, lead: [], byGroup: {},
  pm: null, bandPm: null, range: 10, polarity: 1, shuffled: false, binaural: false, masterMuted: false,
  announceEvery: '5', lastAnnounceT: -1, lastLeadIdx: -1, busy: null, ready: false,
};

// ---------------------------------------------------------------- loading
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

async function boot() {
  $('motion-chk').checked = !reducedMotion;
  if (!canSpeak()) $('speech-chk').disabled = true;
  try {
    const idx = await getJSON('tracks/index.json');
    S.index = (Array.isArray(idx) ? idx : idx.tracks || []).map((t) => (typeof t === 'object' ? t : { id: String(t), title: String(t) }));
  } catch (e) {
    $('track-list').replaceChildren(el('li', { textContent: `Could not load the track list (${e.message}). Serve this folder over HTTP.` }));
    return;
  }
  renderTrackList();
  const want = decodeURIComponent(location.hash.slice(1));
  const first = S.index.find((t) => String(t.id) === want) || S.index[0];
  if (first) loadTrack(first.id, { announce: false });
}

function renderTrackList() {
  const list = $('track-list');
  list.replaceChildren(...S.index.map((t) => {
    const b = el('button', { type: 'button' },
      el('span', { className: 't-title', textContent: t.title }),
      el('span', { className: 't-meta', textContent: [t.voice, t.duration_s ? fmtTime(t.duration_s) : '', `SVS ${t.id}`].filter(Boolean).join(' · ') }));
    b.dataset.id = t.id;
    if (t.summary) b.setAttribute('aria-description', t.summary);
    b.addEventListener('click', () => loadTrack(t.id));
    return el('li', {}, b);
  }));
}

async function loadTrack(id, { announce = true } = {}) {
  engine.pause(); stopBusy();
  S.ready = false; $('play-btn').disabled = true; $('pos').disabled = true;
  for (const b of $('track-list').querySelectorAll('button')) b.setAttribute('aria-current', String(b.dataset.id === String(id)));
  history.replaceState(null, '', `#${encodeURIComponent(id)}`);
  const base = `tracks/${id}/`;
  setStatus('Loading track…');
  let track;
  try {
    track = await getJSON(`${base}track.json`);
    S.timeline = await getJSON(base + (track.timeline || 'timeline.json'));
    S.controls = track.controls ? await getJSON(base + track.controls).catch(() => null) : null;
  } catch (e) { setStatus(`Could not load this track: ${e.message}`, true); return; }
  S.track = track; S.base = base;
  prepareData(track);
  renderTrack(track);
  setPlaying(false);
  if (!engine.supported) { useFallback('This browser has no Web Audio support.'); return; }
  try {
    await engine.load(track, base, (n, total) => setStatus(`Loading audio ${n} of ${total}…`));
  } catch (e) { useFallback(`Audio could not be decoded (${e.message}).`); return; }
  // reset per-track toggles
  S.shuffled = false; S.polarity = 1;
  engine.setShuffled(false); engine.setLiveLead(false);
  syncToggles();
  for (const g of GROUPS) engine.setGroup(g, engine.groupState[g]);
  S.ready = true;
  $('play-btn').disabled = false; $('pos').disabled = false;
  $('pos').max = String(engine.duration);
  $('dur-big').textContent = fmtTime(engine.duration);
  seek(0);
  const msg = `${track.title} loaded, ${spokenTime(engine.duration)}. Press Space to play, L for the legend, question mark for help.`;
  setStatus(msg.replace(/\.\s.*$/, '.') + ' Ready.');
  if (announce) announcer.status(msg);
}

function useFallback(why) {
  const a = $('fallback-audio');
  if (S.track?.mix) { a.src = S.base + S.track.mix; a.hidden = false; }
  setStatus(`${why} Use the basic audio player in the Data source section.`, true);
}

function setStatus(text, error = false) {
  const s = $('load-status'); s.textContent = text; s.classList.toggle('error', error);
  if (error) announcer.status(text);
}

function groupOf(voice) {
  const t = S.track || {};
  if (t.voice_groups?.[voice]) return t.voice_groups[voice];
  if (voice === (t.lead_voice || 'lead')) return 'melody';
  if (voice.startsWith('band')) return 'bands';
  return GROUPS.find((g) => voice.toLowerCase().startsWith(g)) || 'harmony';
}

function prepareData(track) {
  S.timeline.sort((a, b) => a.start - b.start);
  // lead_voices (optional) names the data series the readout speaks; otherwise every melody-group note.
  // Notes without a numeric value (model-written harmony has value null) never become the lead.
  const leadVoices = track.lead_voices?.length ? new Set(track.lead_voices) : null;
  S.lead = S.timeline.filter((n) => (leadVoices ? leadVoices.has(n.voice) : groupOf(n.voice) === 'melody') && Number.isFinite(n.value));
  S.byGroup = {};
  for (const n of S.timeline) (S.byGroup[groupOf(n.voice)] ||= []).push(n);
  const vals = S.lead.map((n) => n.value).filter(Number.isFinite);
  S.vmin = vals.length ? Math.min(...vals) : 0; S.vmax = vals.length ? Math.max(...vals) : 1;
  S.range = (track.pitch_map ? track.pitch_map.vmax - track.pitch_map.vmin : S.vmax - S.vmin) || 10;
  S.pm = track.pitch_map ? new PitchMap(track.pitch_map) : null;
  S.bandPm = track.band_pitch_map ? new PitchMap(track.band_pitch_map) : null;
  // per-band register shift (band_voices in jukebox/score/mapping.py)
  S.bandPms = {};
  for (const v of track.sweep?.voices || []) {
    if (S.bandPm && typeof v === 'object') S.bandPms[v.voice] = new PitchMap({ ...track.band_pitch_map, low: S.bandPm.low + (v.pitch_offset || 0) });
  }
  S.lastLeadIdx = -1; S.lastAnnounceT = -1;
}

// ---------------------------------------------------------------- rendering
function renderTrack(t) {
  $('track-title').textContent = t.title;
  const badge = $('track-voice');
  badge.hidden = !t.voice; badge.textContent = t.voice ? `Voice: ${t.voice}` : '';
  $('track-desc').textContent = t.description || '';
  $('track-caveat').hidden = !t.caveat; $('track-caveat').textContent = t.caveat ? `Note: ${t.caveat}` : '';
  $('units-big').textContent = t.units || '';
  $('value-big').textContent = '–';

  // video: linked from NASA SVS, never copied
  video.replaceChildren();
  video.removeAttribute('src');
  for (const u of [t.video_url, t.video_alt_url].filter(Boolean)) {
    video.append(el('source', { src: u, type: /\.webm$/i.test(u) ? 'video/webm' : 'video/mp4' }));
  }
  video.hidden = !t.video_url;
  video.load();
  // a still image (e.g. 5669 aerosols) is swept west to east: a line marks the column being played
  $('still-wrap').hidden = !t.image_url || !!t.video_url;
  if (t.image_url && !t.video_url) { still.src = t.image_url; still.alt = `NASA SVS image: ${t.title}`; } else still.removeAttribute('src');
  video.closest('figure').hidden = !t.video_url && !t.image_url;
  $('video-note').textContent = t.video_url
    ? 'Video streams from NASA SVS and has no sound; the music plays in sync with it.'
    : t.image_url ? 'Still image from NASA SVS. The music sweeps across it from west to east; the line shows where.'
      : 'No video for this track.';

  // legend
  $('legend').replaceChildren(...(t.legend || []).flatMap((l) => [el('dt', { textContent: l.voice }), el('dd', { textContent: l.meaning })]));
  $('pitch-range').textContent = S.pm
    ? `Lead pitch: ${noteName(S.pm.pitch(S.pm.vmin))} = ${fmtValue(S.pm.vmin, t.units, S.range)}, up to ${noteName(S.pm.pitch(S.pm.vmax))} = ${fmtValue(S.pm.vmax, t.units, S.range)}. Press L to hear it.`
    : '';

  renderFidelity(t);
  renderMixer(t);
  renderNotes(t);

  $('data-source').textContent = typeof t.data_source === 'string' ? t.data_source : JSON.stringify(t.data_source || '');
  $('svs-link').href = t.svs_url || '#';
  $('svs-link').textContent = `NASA SVS page ${t.id}`;
  const mix = $('mix-link');
  mix.hidden = !t.mix; if (t.mix) mix.href = S.base + t.mix;
  $('fallback-audio').hidden = true;

  // feature availability
  $('ab-btn').disabled = !t.control;
  $('polarity-btn').disabled = !S.pm;
  $('sweep-btn').disabled = !sweepVoices().length;
  $('overview-btn').disabled = !S.lead.length;
  roll.setData({ notes: S.timeline, duration: t.duration_s || Math.max(...S.timeline.map((n) => n.end)), groupOf, controls: S.controls });
}

function renderFidelity(t) {
  const f = t.fidelity || {};
  if (f.r2 == null) {
    const series = Object.entries(f.detail || {}).map(([k, v]) => [k, v.final_stem]).filter(([, r]) => r?.pitch_within_half_semitone != null);
    $('fid-text').textContent = series.length
      ? 'This track has no single lead melody. Each series was played back into a pitch tracker on its own: ' +
        series.map(([k, r]) => `${k}, ${Math.round(r.pitch_within_half_semitone * 100)}% of ${r.n} notes on the right pitch`).join('; ') + '.'
      : 'No fidelity check for this track yet.';
    $('fid-bars').replaceChildren(); $('fid-detail').textContent = ''; return;
  }
  const follow = f.r2 >= 0.9 ? 'follows the data closely' : f.r2 >= 0.6 ? 'follows the data fairly well' : 'follows the data only loosely';
  const ctrl = f.control_r2 == null ? '' : f.control_r2 < 0.2 ? '; shuffled data does not.' : '; shuffled data scores almost as well, so treat this track with care.';
  $('fid-text').textContent = `The melody ${follow}${ctrl || '.'}`;
  const pct = (r) => `${Math.round(Math.max(0, r) * 100)}%`;
  const share = (r) => (r <= 0.005 ? 'none of them' : pct(r));
  const units = t.units || '';
  $('fid-detail').textContent =
    `We played the lead back into a pitch tracker and compared it with the data. The pitch explains ${pct(f.r2)} of the ups and downs ` +
    `(R² ${fmtNum(f.r2)}${f.mae != null ? `, average error ${fmtValue(f.mae, units, 1)}` : ''}).` +
    (f.control_r2 != null ? ` The same notes in shuffled time order explain ${share(f.control_r2)} (R² ${fmtNum(f.control_r2)}). Press A to hear the difference.` : '');
  const bar = (label, r, cls = '') => el('div', { className: 'fid-bar' },
    el('span', { textContent: label }),
    el('div', { className: 'fid-track' }, Object.assign(el('div', { className: `fid-fill ${cls}` }), { style: `width:${pct(r)}` })),
    el('span', { textContent: fmtNum(r) }));
  if (f.mix_r2 != null) $('fid-detail').textContent += ` Measured on the full mix (harmony and Earth sounds included), the pitch still explains ${pct(f.mix_r2)} (R² ${fmtNum(f.mix_r2)}).`;
  $('fid-bars').replaceChildren(bar('Real data', f.r2), ...(f.mix_r2 != null ? [bar('Full mix', f.mix_r2)] : []),
    ...(f.control_r2 != null ? [bar('Shuffled', f.control_r2, 'ctrl')] : []));
}

function renderMixer(t) {
  const present = new Set((t.stems || []).map((s) => s.group));
  if (t.control) present.add(t.control.replaces || 'melody');
  const items = GROUPS.map((g, i) => {
    if (!present.has(g)) return null;
    const st = engine.groupState[g];
    const id = `mute-${g}`;
    const chk = el('input', { type: 'checkbox', id, checked: !st.muted });
    chk.setAttribute('aria-keyshortcuts', String(i + 1));
    chk.addEventListener('change', () => { engine.setGroup(g, { muted: !chk.checked }); });
    const vol = el('input', { type: 'range', min: '0', max: '1.5', step: '0.05', value: String(st.volume) });
    vol.setAttribute('aria-label', `${GROUP_LABELS[g]} volume`);
    const setText = () => vol.setAttribute('aria-valuetext', `${Math.round(vol.value * 100)} percent`);
    setText();
    vol.addEventListener('input', () => { engine.setGroup(g, { volume: +vol.value }); setText(); });
    const stems = (t.stems || []).filter((s) => s.group === g).map((s) => s.label || s.name).join(', ');
    const li = el('li', {},
      chk, el('label', { htmlFor: id }, el('kbd', { textContent: String(i + 1) }), ` ${GROUP_LABELS[g]}`, stems ? el('span', { className: 'sr', textContent: ` (${stems})` }) : ''),
      el('div', { className: 'vol' }, vol));
    li.style.setProperty('--c', `var(--${g})`);
    return li;
  }).filter(Boolean);
  const absent = GROUPS.filter((g) => !present.has(g)).map((g) => GROUP_LABELS[g].split(' ')[0]);
  if (absent.length) items.push(el('li', { className: 'absent', textContent: `Not in this track: ${absent.join(', ')}.` }));
  $('mixer').replaceChildren(...items);
}

function renderNotes(t) {
  const u = t.units || '';
  $('note-table').tBodies[0].replaceChildren(...S.lead.map((n, i) => el('tr', {},
    el('td', { textContent: String(i + 1) }), el('td', { textContent: fmtTime(n.start) }),
    el('td', { textContent: fmtValue(n.value, u, S.range) }), el('td', { textContent: noteName(n.pitch) }))));
  if (!S.lead.length) { $('roll-text').textContent = 'This track has no lead melody.'; return; }
  const lo = S.lead.reduce((a, b) => (b.value < a.value ? b : a)), hi = S.lead.reduce((a, b) => (b.value > a.value ? b : a));
  const first = S.lead[0], last = S.lead[S.lead.length - 1];
  const mid = S.lead.length >> 1;
  const avg = (a) => a.reduce((s, n) => s + n.value, 0) / a.length;
  const d = avg(S.lead.slice(mid)) - avg(S.lead.slice(0, mid));
  const trend = Math.abs(d) < 0.1 * (S.vmax - S.vmin) ? 'no clear overall trend' : d > 0 ? 'higher on average in the second half' : 'lower on average in the second half';
  const others = Object.keys(S.byGroup).filter((g) => g !== 'melody');
  $('roll-text').textContent =
    `Picture of the score: ${S.lead.length} lead notes over ${spokenTime(t.duration_s)}. ` +
    `The value starts at ${fmtValue(first.value, u, S.range)} and ends at ${fmtValue(last.value, u, S.range)}; ` +
    `lowest ${fmtValue(lo.value, u, S.range)} at ${fmtTime(lo.start)}, highest ${fmtValue(hi.value, u, S.range)} at ${fmtTime(hi.start)}; ${trend}.` +
    (others.length ? ` Background voices drawn fainter: ${others.join(', ')}.` : '') +
    (S.controls?.synthetic ? ' The voice-balance strip under the notes is a placeholder.' : '');
}

// ---------------------------------------------------------------- time, values
function leadIndexAt(t) {
  let lo = 0, hi = S.lead.length - 1, ans = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (S.lead[m].start <= t + 1e-6) { ans = m; lo = m + 1; } else hi = m - 1; }
  return ans;
}
const noteAt = (notes, t) => {
  let best = null;
  for (const n of notes) { if (n.start <= t + 1e-6) { if (t < n.end) return n; best = n; } else break; }
  return best;
};

const ymd = (s) => (/^\d{8}$/.test(s) ? `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}` : String(s));
function dateLabel(iso) {
  if (/^\d{4}$/.test(iso)) return iso;
  const d = new Date(`${ymd(iso)}T12:00:00Z`);
  return Number.isNaN(+d) ? String(iso) : d.toLocaleDateString(undefined, { dateStyle: 'medium', timeZone: 'UTC' });
}
function segmentAt(t) {
  const seg = (S.track?.segments || []).find((s) => t >= s.start && t < s.end);
  return seg ? `${seg.name}, ` : '';
}
function timeLabel(t) {
  const tr = S.track || {};
  const cal = tr.calendar;
  if (cal?.ref_date && cal.sec_per_year) {   // e.g. 5011: date = ref_date + (t - t_ref) / sec_per_year years, clamped
    let ms = Date.parse(`${cal.ref_date}T00:00:00Z`) + ((t - cal.t_ref) / cal.sec_per_year) * 365.2425 * 864e5;
    if (cal.clamp) ms = Math.min(Math.max(ms, Date.parse(`${cal.clamp[0]}T00:00:00Z`)), Date.parse(`${cal.clamp[1]}T00:00:00Z`));
    return new Date(ms).toLocaleDateString(undefined, { year: 'numeric', month: 'long', timeZone: 'UTC' });
  }
  if (tr.dates?.length) {   // dates_t: explicit anchor times; otherwise evenly spread over the track
    let i;
    if (tr.dates_t?.length === tr.dates.length) { i = -1; for (let k = 0; k < tr.dates_t.length && tr.dates_t[k] <= t + 1e-6; k++) i = k; }
    else i = Math.floor((t / (tr.duration_s || engine.duration || 1)) * tr.dates.length);
    if (i >= 0) return segmentAt(t) + dateLabel(tr.dates[Math.min(i, tr.dates.length - 1)]);
  }
  const ax = tr.time_axis;
  if (ax?.start && ax?.end) {
    const a = Date.parse(ax.start), b = Date.parse(ax.end);
    const d = new Date(a + (b - a) * (t / (engine.duration || 1)));
    return d.toLocaleDateString(undefined, ax.format === 'year' ? { year: 'numeric' } : ax.format === 'day' ? { dateStyle: 'medium' } : { year: 'numeric', month: 'long' });
  }
  return segmentAt(t) + spokenTime(t);
}

function sweepFraction(t) {
  const sw = S.track?.sweep;
  return Math.min(1, Math.max(0, t / (sw?.duration_s || S.track?.duration_s || engine.duration || 1)));
}

function describeNow(t, { full = false } = {}) {
  const u = S.track?.units || '';
  const i = leadIndexAt(t);
  const sw = S.track?.sweep;
  if (!S.lead.length && sw?.lon_start != null) {
    const lon = sw.lon_start + sweepFraction(t) * (sw.lon_end - sw.lon_start);
    return `${spokenTime(t)}: sweep at longitude ${Math.abs(Math.round(lon))} degrees ${lon < 0 ? 'west' : 'east'}.`;
  }
  if (i < 0) return `${timeLabel(t)}. No data yet.`;
  const n = S.lead[i], prev = S.lead[i - 1];
  let msg = `${timeLabel(t)}: ${spokenValue(n.value, u, S.range)}`;
  if (prev && full) msg += n.value > prev.value + 1e-9 ? ', rising' : n.value < prev.value - 1e-9 ? ', falling' : ', steady';
  if (S.shuffled) msg += '. Shuffled control is playing';
  if (full && S.controls && !S.controls.synthetic) {
    const k = S.controls.t.findIndex((x) => x >= t);
    const j = k < 0 ? S.controls.t.length - 1 : k;
    const parts = ['geophony', 'anthrophony', 'biophony'].map((v) => [v, S.controls.curves[v]?.[j]]).filter(([, x]) => x > 0.05);
    if (parts.length) msg += `. Voices: ${parts.map(([v, x]) => `${v} ${Math.round(x * 100)} percent`).join(', ')}`;
  }
  return `${msg}.`;
}

// ---------------------------------------------------------------- transport
function setPlaying(p) {
  const b = $('play-btn');
  b.textContent = p ? 'Pause' : 'Play';
  b.setAttribute('aria-label', p ? 'Pause' : 'Play');
}

async function togglePlay() {
  if (!S.ready) return;
  stopBusy();
  if (engine.playing) {
    engine.pause(); video.pause(); setPlaying(false);
    announcer.status(`Paused at ${describeNow(engine.time)}`);
  } else {
    await engine.play(); setPlaying(true);
    syncVideo(true);
    S.lastAnnounceT = engine.time;
  }
}

function seek(t, announce = false) {
  if (!S.ready) return;
  engine.seek(t);
  syncVideo(true);
  updateReadout(engine.time, true);
  S.lastLeadIdx = leadIndexAt(engine.time);
  if (announce) announcer.say(describeNow(engine.time), { minGap: 0.4 });
}

function syncVideo(force = false) {
  if (!video.currentSrc && !video.querySelector('source')) return;
  const rate = (S.track && S.track.video_rate) || 1;   // < 1: music slower than the movie (5046)
  const t = engine.time * rate;
  if (video.playbackRate !== rate) video.playbackRate = rate;
  const animate = $('motion-chk').checked;
  if (engine.playing && animate) {
    if (video.paused) video.play().catch(() => {});
    if (force || Math.abs(video.currentTime - t) > 0.3) try { video.currentTime = t; } catch { /* not seekable yet */ }
  } else {
    if (!video.paused) video.pause();
    if (force) try { video.currentTime = t; } catch { /* */ }
  }
}

function updateReadout(t, forceSlider = false) {
  $('time-big').textContent = fmtTime(t);
  const i = leadIndexAt(t);
  $('value-big').textContent = i >= 0 ? fmtValue(S.lead[i].value, '', S.range) : '–';
  const pos = $('pos');
  if (forceSlider || document.activeElement !== pos) {
    pos.value = String(t);
    pos.setAttribute('aria-valuetext', `${fmtTime(t)} of ${fmtTime(engine.duration)}${i >= 0 ? `, ${spokenValue(S.lead[i].value, S.track.units, S.range)}` : ''}`);
  }
}

function stepNote(dir) {
  if (!S.ready || !S.lead.length) return;
  const cur = leadIndexAt(engine.time);
  const target = Math.max(0, Math.min(S.lead.length - 1, cur + dir));
  const n = S.lead[target];
  seek(n.start);
  if (!engine.playing) { engine.ensure(); engine.ctx.resume(); }
  if (!engine.playing) engine.tone(pitchOf(n), { dur: Math.min(0.6, n.end - n.start), vel: 95 });
  const prev = S.lead[target - 1];
  const change = !prev ? '' : n.value > prev.value ? ', up' : n.value < prev.value ? ', down' : ', same';
  announcer.say(`Note ${target + 1} of ${S.lead.length}, ${fmtTime(n.start)}: ${spokenValue(n.value, S.track.units, S.range)}${change}.`);
}

const pitchOf = (n) => (S.polarity < 0 && S.pm ? S.pm.pitch(n.value, -1) : n.pitch);
const bandPitchOf = (n) => {
  const pm = S.bandPms[n.voice] || S.bandPm;
  return S.polarity < 0 && pm ? pm.pitch(n.value, -1) : n.pitch;
};

// ---------------------------------------------------------------- the loop
function frame() {
  const t = engine.time;
  updateReadout(t);
  roll.draw(t);
  if (!$('still-wrap').hidden) $('sweep-line').style.left = `${(sweepFraction(t) * 100).toFixed(2)}%`;
  requestAnimationFrame(frame);
}

function tick() {
  if (!S.ready) return;
  engine.tick();
  const t = engine.time;
  if (engine.playing) {
    if (t >= engine.duration - 0.02) {
      engine.pause(); engine.seek(engine.duration); video.pause(); setPlaying(false);
      announcer.status('End of track. Press Space to play again from the start.');
      return;
    }
    syncVideo();
    if (S.binaural) driveBinaural(t);
    autoAnnounce(t);
  }
}

function autoAnnounce(t) {
  const mode = S.announceEvery;
  if (mode === '0') return;
  if (mode === 'note') {
    const i = leadIndexAt(t);
    if (i !== S.lastLeadIdx) { S.lastLeadIdx = i; announcer.say(describeNow(t), { minGap: 1.5 }); }
    return;
  }
  if (t - S.lastAnnounceT >= +mode) { S.lastAnnounceT = t; announcer.say(describeNow(t), { minGap: 2 }); }
}

function driveBinaural(t) {
  for (const [g, notes] of Object.entries(S.byGroup)) {
    let sum = 0, k = 0;
    for (const n of notes) { if (n.start > t) break; if (t < n.end) { sum += n.pan || 0; k++; } }
    if (k) engine.setGroupPan(g, Math.round((sum / k) * 20) / 20);
  }
}

// ---------------------------------------------------------------- listening tools
function pausePlayback() {
  if (engine.playing) { engine.pause(); video.pause(); setPlaying(false); }
}

function stopBusy() {
  if (S.busy) { S.busy.cancelled = true; S.busy = null; }
  cancelSpeech();
  engine.duck(1);
}
function startBusy() { stopBusy(); S.busy = { cancelled: false }; return S.busy; }

async function legend() {
  if (!S.track) return;
  if (S.busy?.kind === 'legend') { stopBusy(); announcer.status('Legend stopped.'); return; }
  pausePlayback();
  const job = startBusy(); job.kind = 'legend';
  engine.ensure(); await engine.ctx.resume();
  const t = S.track, u = t.units || '';
  const steps = [`Legend for ${t.title}.`, ...(t.legend || []).map((l) => `${l.voice}: ${l.meaning}`)];
  $('legend').scrollIntoView?.({ block: 'nearest', behavior: reducedMotion ? 'auto' : 'smooth' });
  // Browser speech keeps words and tones in order; without it, fall back to the live region.
  const talk = (s) => { if (!canSpeak()) announcer.say(s); return speak(s); };
  for (const s of steps) { if (job.cancelled) return; await talk(s); }
  const cal = [[S.pm, 'Lead']];
  if (S.bandPm && sweepVoices().length) cal.push([S.bandPm, 'Bands']);
  for (const [pm, name] of cal) {
    if (!pm || job.cancelled) continue;
    await talk(`${name} calibration tones${S.polarity < 0 ? ', polarity flipped' : ''}.`);
    for (const [label, v] of [['Lowest', pm.vmin], ['Middle', pm.vmid], ['Highest', pm.vmax]]) {
      if (job.cancelled) return;
      await talk(`${label}: ${spokenValue(v, u, pm.vmax - pm.vmin)}.`);
      if (job.cancelled) return;
      engine.tone(pm.pitch(v, S.polarity), { dur: 0.8, vel: 100, timbre: 'pluck' });
      await sleep(1100);
    }
  }
  if (!job.cancelled) { await talk('End of legend. Press Space to play.'); S.busy = null; }
}

function sweepVoices() {
  const t = S.track;
  if (!t) return [];
  if (t.sweep?.voices) return t.sweep.voices.map((v) => (typeof v === 'string' ? { voice: v, label: v } : v));
  const vs = [...new Set(S.timeline.filter((n) => groupOf(n.voice) === 'bands').map((n) => n.voice))];
  return vs.length > 1 ? vs.map((v, i) => ({ voice: v, label: v, pan: vs.length > 1 ? -1 + (2 * i) / (vs.length - 1) : 0 })) : [];
}

async function sweep() {
  const voices = sweepVoices();
  if (!voices.length) { announcer.status('Sweep is not available for this track.'); return; }
  const job = startBusy(); job.kind = 'sweep';
  engine.ensure(); await engine.ctx.resume();
  const t = engine.time, u = S.track.units || '';
  const hits = voices.map((v) => ({ ...v, n: noteAt(S.timeline.filter((n) => n.voice === v.voice), t) })).filter((h) => h.n);
  engine.duck(0.2);
  let when = engine.ctx.currentTime + 0.05;
  for (const h of hits) { engine.tone(bandPitchOf(h.n), { when, dur: 0.32, vel: 100, pan: h.pan ?? 0 }); when += 0.42; }
  await sleep((when - engine.ctx.currentTime) * 1000 + 300);
  if (job.cancelled) return;
  engine.duck(1); S.busy = null;
  const range = S.bandPm ? S.bandPm.vmax - S.bandPm.vmin : S.range;
  announcer.say(`Sweep, ${S.track.sweep?.label || 'bands'}, at ${fmtTime(t)}: ` + hits.map((h) => `${h.label} ${spokenValue(h.n.value, u, range)}`).join('; ') + '.');
}

async function overview() {
  if (!S.lead.length) return;
  pausePlayback();
  const job = startBusy(); job.kind = 'overview';
  engine.ensure(); await engine.ctx.resume();
  const u = S.track.units || '';
  const lo = S.lead.reduce((a, b) => (b.value < a.value ? b : a)), hi = S.lead.reduce((a, b) => (b.value > a.value ? b : a));
  announcer.say('Overview: the whole lead in 8 seconds, from left to right.');
  await sleep(2500);
  if (job.cancelled) return;
  const dur = S.track.duration_s || engine.duration, k = 8 / dur, t0 = engine.ctx.currentTime + 0.05;
  for (const n of S.lead) {
    engine.tone(pitchOf(n), { when: t0 + n.start * k, dur: Math.max(0.07, (n.end - n.start) * k * 0.9), vel: n.velocity, pan: -0.9 + 1.8 * (n.start / dur) });
  }
  await sleep(8400);
  if (job.cancelled) return;
  S.busy = null;
  announcer.say(`End of overview. Lowest ${spokenValue(lo.value, u, S.range)} at ${fmtTime(lo.start)}; highest ${spokenValue(hi.value, u, S.range)} at ${fmtTime(hi.start)}.`);
}

function toggleAB() {
  if (!S.track?.control) { announcer.status('This track has no shuffled control.'); return; }
  if (S.polarity < 0) togglePolarity(true);
  S.shuffled = !S.shuffled;
  engine.setShuffled(S.shuffled);
  syncToggles();
  const f = S.track.fidelity || {};
  announcer.status(S.shuffled
    ? `B: shuffled control. The same notes in random time order${f.control_r2 != null ? `, R squared ${spokenNum(f.control_r2)}` : ''}.`
    : `A: real data${f.r2 != null ? `, R squared ${spokenNum(f.r2)}` : ''}.`);
}

function togglePolarity(quiet = false) {
  if (!S.pm) { announcer.status('Polarity flip is not available for this track.'); return; }
  if (S.shuffled && S.polarity > 0) { S.shuffled = false; engine.setShuffled(false); }
  S.polarity = -S.polarity;
  engine.setLiveLead(S.polarity < 0, S.lead, pitchOf);
  syncToggles();
  if (!quiet) announcer.status(S.polarity < 0
    ? 'Polarity flipped: higher values now play lower. The lead is synthesized live from the data.'
    : 'Normal polarity: higher pitch means a higher value. Recorded lead restored.');
}

function toggleBinaural() {
  S.binaural = !S.binaural;
  engine.ensure();
  engine.setBinaural(S.binaural);
  if (S.binaural) driveBinaural(engine.time);
  syncToggles();
  announcer.status(S.binaural ? 'Headphone 3D sound on. Use headphones.' : 'Headphone 3D sound off. Normal stereo.');
}

function toggleMasterMute() {
  S.masterMuted = !S.masterMuted;
  engine.ensure(); engine.setMasterMuted(S.masterMuted);
  syncToggles();
  announcer.status(S.masterMuted ? 'All sound muted.' : 'Sound on.');
}

function toggleGroup(i) {
  const g = GROUPS[i];
  const chk = $(`mute-${g}`);
  if (!chk) { announcer.status(`No ${GROUP_LABELS[g]} in this track.`); return; }
  chk.checked = !chk.checked;
  engine.setGroup(g, { muted: !chk.checked });
  announcer.status(`${GROUP_LABELS[g]} ${chk.checked ? 'on' : 'muted'}.`);
}

function syncToggles() {
  $('ab-btn').setAttribute('aria-pressed', String(S.shuffled));
  $('polarity-btn').setAttribute('aria-pressed', String(S.polarity < 0));
  $('hrtf-btn').setAttribute('aria-pressed', String(S.binaural));
  $('mute-btn').setAttribute('aria-pressed', String(S.masterMuted));
}

function sayValue() {
  if (!S.track) return;
  const text = describeNow(engine.time, { full: true });
  announcer.say(text);
}

function openHelp() {
  const d = $('help');
  if (d.open) d.close(); else if (d.showModal) d.showModal(); else d.setAttribute('open', '');
}

// ---------------------------------------------------------------- input
const on = (id, fn) => $(id).addEventListener('click', fn);
on('play-btn', togglePlay);
on('back-btn', () => seek(engine.time - 5, true));
on('fwd-btn', () => seek(engine.time + 5, true));
on('prev-note-btn', () => stepNote(-1));
on('next-note-btn', () => stepNote(1));
on('legend-btn', legend);
on('value-btn', sayValue);
on('sweep-btn', sweep);
on('overview-btn', overview);
on('ab-btn', toggleAB);
on('polarity-btn', () => togglePolarity());
on('hrtf-btn', toggleBinaural);
on('mute-btn', toggleMasterMute);
on('help-btn', openHelp);
$('pos').addEventListener('input', (e) => seek(+e.target.value, true));
$('announce-sel').addEventListener('change', (e) => { S.announceEvery = e.target.value; });
$('speech-chk').addEventListener('change', (e) => { announcer.speech = e.target.checked; if (!e.target.checked) cancelSpeech(); });
$('motion-chk').addEventListener('change', () => syncVideo(true));
video.addEventListener('error', () => { $('video-note').textContent = 'The NASA video could not be loaded; the music still plays.'; }, true);
window.addEventListener('hashchange', () => {
  const id = decodeURIComponent(location.hash.slice(1));
  if (id && id !== String(S.track?.id) && S.index.some((t) => String(t.id) === id)) loadTrack(id);
});

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const tgt = e.target;
  const tag = tgt.tagName;
  const helpOpen = $('help').open;
  if (e.key === '?') { e.preventDefault(); openHelp(); return; }
  if (helpOpen) return;
  if (tag === 'SELECT' || tag === 'TEXTAREA' || (tag === 'INPUT' && !['checkbox', 'range'].includes(tgt.type))) return;
  if (tgt.type === 'range' && ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown'].includes(e.key)) return;
  const activates = ['BUTTON', 'SUMMARY', 'A'].includes(tag) || tgt.type === 'checkbox';
  if ((e.key === ' ' || e.key === 'Enter') && activates) return; // let the focused control handle it
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  const act = {
    ' ': togglePlay,
    ArrowLeft: () => (e.shiftKey ? stepNote(-1) : seek(engine.time - 5, true)),
    ArrowRight: () => (e.shiftKey ? stepNote(1) : seek(engine.time + 5, true)),
    Home: () => seek(0, true),
    l: legend, v: sayValue, s: () => (e.shiftKey ? overview() : sweep()),
    p: () => togglePolarity(), a: toggleAB, h: toggleBinaural, m: toggleMasterMute,
    Escape: () => { if (S.busy) { stopBusy(); announcer.status('Stopped.'); } },
  }[k] || (/^[1-7]$/.test(e.key) ? () => toggleGroup(+e.key - 1) : null);
  if (!act) return;
  e.preventDefault();
  act();
});

setInterval(tick, 100);
requestAnimationFrame(frame);
boot();
