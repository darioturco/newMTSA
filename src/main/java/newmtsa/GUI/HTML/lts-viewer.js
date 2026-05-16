function spreadParallelEdges(edges) {
  const grouped = new Map();
  for (const e of edges) {
    const k = e.from + '->' + e.to;
    if (!grouped.has(k)) grouped.set(k, []);
    grouped.get(k).push(e);
  }
  for (const list of grouped.values()) {
    if (list.length <= 1) continue;
    const mid = (list.length - 1) / 2;
    for (let i = 0; i < list.length; i++) {
      const off = i - mid;
      list[i].smooth = {
        enabled: true,
        type: off >= 0 ? 'curvedCW' : 'curvedCCW',
        roundness: 0.12 + Math.abs(off) * 0.1
      };
    }
  }
  return edges;
}

function switchTab(name) {
  document.getElementById('ltsTab').style.display = name === 'lts' ? '' : 'none';
  document.getElementById('traceTab').style.display = name === 'trace' ? '' : 'none';
  document.getElementById('tabLts').classList.toggle('active', name === 'lts');
  document.getElementById('tabTrace').classList.toggle('active', name === 'trace');
  if (name === 'trace') initTraceTabIfNeeded();
}

const ltsSelect = document.getElementById('ltsSelect');
const actionsList = document.getElementById('actionsList');
const ltsContainer = document.getElementById('network');
let ltsNetwork = null;

const ltsOpts = {
  layout: { improvedLayout: true },
  nodes: {
    font: { size: 14 },
    borderWidth: 1,
    margin: 12
  },
  groups: {
    initial: { color: { background: '#bfdbfe', border: '#2563eb' }, borderWidth: 3 },
    marked:  { color: { background: '#bbf7d0', border: '#16a34a' }, borderWidth: 3 },
    error:   { color: { background: '#fee2e2', border: '#dc2626' }, borderWidth: 3 },
    normal:  { color: { background: '#ffffff', border: '#9ca3af' }, borderWidth: 1 }
  },
  edges: {
    font: {
      align: 'top',
      size: 12,
      background: 'rgba(255,255,255,0.8)'
    },
    smooth: { enabled: true, type: 'dynamic' }
  },
  physics: {
    stabilization: true,
    barnesHut: {
      springLength: 240,
      springConstant: 0.03,
      gravitationalConstant: -5000,
      damping: 0.18
    }
  }
};

function renderActionsList(data) {
  actionsList.innerHTML = '';
  if (!data?.actions?.length) {
    actionsList.innerHTML = '<span style="color:#6b7280;font-size:13px">(no actions)</span>';
    return;
  }
  for (const a of data.actions) {
    const el = document.createElement('code');
    el.textContent = a;
    actionsList.appendChild(el);
  }
}

function renderSelected() {
  const key = ltsSelect.value;
  const data = graphs[key];
  if (!data) return;
  renderActionsList(data);
  const warn = document.getElementById('renderWarning');
  if (data.blocked) {
    warn.textContent = `Rendering blocked: ${data.stateCount} states (max ${MAX_STATE_RENDER}).`;
    warn.style.display = 'block';
    const empty = { nodes: new vis.DataSet([]), edges: new vis.DataSet([]) };
    if (ltsNetwork === null) {
      ltsNetwork = new vis.Network(ltsContainer, empty, ltsOpts);
    } else {
      ltsNetwork.setData(empty);
    }
    return;
  }
  warn.style.display = 'none';
  const vd = {
    nodes: new vis.DataSet(data.nodes),
    edges: new vis.DataSet(spreadParallelEdges([...data.edges]))
  };
  if (ltsNetwork === null) {
    ltsNetwork = new vis.Network(ltsContainer, vd, ltsOpts);
  } else {
    ltsNetwork.setData(vd);
  }
  ltsNetwork.fit({ animation: true });
}

function downloadSelectedDot() {
  const key = ltsSelect.value;
  const data = graphs[key];
  if (!data?.dot) return;
  const blob = new Blob([data.dot], { type: 'text/vnd.graphviz' });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement('a'), {
    href: url,
    download: key.replace(/[<>:"/\\|?*]+/g, '_') + '.dot'
  });
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

ltsSelect.addEventListener('change', renderSelected);
document.getElementById('downloadDotBtn').addEventListener('click', downloadSelectedDot);
renderSelected();

let currentStep = 0;
let traceNetwork = null;
let currentNodesDS = null;
let savedPositions = {};
let playInterval = null;
let traceInited = false;

const UNKNOWN_ID = '__unknown__';

const traceNetOpts = {
  layout: { improvedLayout: true },
  nodes: {
    font: { size: 12 },
    borderWidth: 2,
    shadow: false,
    size: 22,
    chosen: false
  },
  edges: {
    font: {
      size: 11,
      align: 'top',
      background: 'rgba(255,255,255,0.85)'
    },
    smooth: { enabled: true, type: 'straightCross' },
    arrows: {
      to: { enabled: true, scaleFactor: 0.8 }
    }
  },
  physics: {
    enabled: true,
    barnesHut: {
      springLength: 220,
      springConstant: 0.04,
      gravitationalConstant: -9000,
      damping: 0.25,
      centralGravity: 0,
      avoidOverlap: 1
    },
    stabilization: {
      iterations: 350,
      fit: true
    }
  }
};

let showIds = true;
let hideUnknown = true;
let showDirector = false;
let selectedNodeId = null;
const nodeIdMap = new Map();
let nextDisplayId = 0;

function getDisplayLabel(state) {
  if (!showIds) return state;
  if (!nodeIdMap.has(state)) {
    nodeIdMap.set(state, nextDisplayId++);
  }
  return String(nodeIdMap.get(state));
}

function toggleLabels() {
  showIds = !showIds;
  document.getElementById('toggleLabelsBtn').textContent = showIds ? 'Show Names' : 'Show IDs';
  if (traceNetwork !== null) {
    renderTraceAtStep(currentStep);
  }
}

function flashRed(el) {
  el.classList.add('animate-flash');
  el.addEventListener('animationend', () => el.classList.remove('animate-flash'), { once: true });
}

function toggleDirector() {
  showDirector = !showDirector;
  const btn = document.getElementById('directorBtn');
  btn.textContent = showDirector ? 'Hide Director' : 'Show Director';
  const dirEdges = traceData && traceData.directorEdges || [];
  if (dirEdges.length === 0) return;
  if (traceNetwork !== null) {
    renderTraceAtStep(currentStep);
  }
}

function toggleUnknown() {
  hideUnknown = !hideUnknown;
  document.getElementById('hideUnknownBtn').textContent = hideUnknown ? 'Show Unknown' : 'Hide Unknown';
  if (traceNetwork !== null) {
    renderTraceAtStep(currentStep);
  }
}

function bfsDistances(initState, edgeMap, expanded) {
  const dist = new Map();
  if (!initState) return dist;
  dist.set(initState, 0);
  const q = [initState];
  while (q.length > 0) {
    const curr = q.shift();
    const d = dist.get(curr);
    for (const t of edgeMap.values()) {
      if (t.from === curr && expanded.has(t.to) && !dist.has(t.to)) {
        dist.set(t.to, d + 1);
        q.push(t.to);
      }
    }
  }
  for (const s of expanded) {
    if (!dist.has(s)) dist.set(s, 0);
  }
  return dist;
}

function buildGraphAtStep(step) {
  const steps = traceData.steps;
  const N = steps.length;
  const ctrlSet = new Set(traceData.controllable);
  const initState = (N > 0 && steps[0].frontier.length > 0) ? steps[0].frontier[0].from : null;
  const expanded = new Set();
  if (initState) expanded.add(initState);
  for (let i = 0; i < step; i++) {
    expanded.add(steps[i].selected.to);
  }
  for (let i = 0; i <= step && i < N; i++) {
    for (const t of steps[i].frontier) {
      expanded.add(t.from);
    }
  }
  const newlyDisc = step > 0 ? steps[step - 1].selected.to : null;
  const edgeMap = new Map();
  for (let i = 0; i <= step && i < N; i++) {
    for (const t of steps[i].frontier) {
      const k = t.from + '\x01' + t.action + '\x01' + t.to;
      if (!edgeMap.has(k)) {
        edgeMap.set(k, t);
      }
    }
  }
  const frontierKeys = new Set();
  let selectedKey = null;
  if (step < N) {
    for (const t of steps[step].frontier) {
      frontierKeys.add(t.from + '\x01' + t.action + '\x01' + t.to);
    }
    const sel = steps[step].selected;
    selectedKey = sel.from + '\x01' + sel.action + '\x01' + sel.to;
  }
  const outgoingCount = new Map();
  for (const t of edgeMap.values()) {
    outgoingCount.set(t.from, (outgoingCount.get(t.from) || 0) + 1);
  }
  const dist = bfsDistances(initState, edgeMap, expanded);
  const maxDist = dist.size > 0 ? Math.max(...dist.values()) : 0;
  const X_SPACING = 260;
  const layerIdx = new Map();
  const usedPos = new Set();
  let hasUnknown = false;
  const nodesArr = [];
  function isMarkedState(s) {
    const hash = s.lastIndexOf('#');
    if (hash === -1) return false;
    const pipe = s.lastIndexOf('|', hash);
    if (pipe === -1) return false;
    return s.substring(pipe + 1, hash) === 'on';
  }
  for (const state of expanded) {
    let bg, border, bw;
    const isError = !outgoingCount.has(state);
    if (state === initState) {
      bg = '#bfdbfe';
      border = '#2563eb';
      bw = 3;
    } else if (isMarkedState(state)) {
      bg = '#bbf7d0';
      border = '#16a34a';
      bw = 2;
    } else if (isError) {
      bg = '#fee2e2';
      border = '#dc2626';
      bw = 2;
    } else if (state === newlyDisc) {
      bg = '#fef3c7';
      border = '#d97706';
      bw = 2;
    } else {
      bg = '#f9fafb';
      border = '#9ca3af';
      bw = 1.5;
    }
    if (state === selectedNodeId) {
      border = '#2563eb';
      bw = 3;
    }
    const pos = savedPositions[state];
    let nodeExtra;
    if (pos) {
      let y = pos.y;
      while (usedPos.has(`${pos.x},${y}`)) y += 100;
      nodeExtra = { x: pos.x, y, fixed: { x: true, y: true } };
    } else {
      const d = dist.get(state) ?? 0;
      const cnt = layerIdx.get(d) || 0;
      layerIdx.set(d, cnt + 1);
      const x = d * X_SPACING;
      let y = cnt * 100;
      while (usedPos.has(`${x},${y}`)) y += 100;
      nodeExtra = { x, y, fixed: { x: true, y: false } };
    }
    usedPos.add(`${nodeExtra.x},${nodeExtra.y}`);
    nodesArr.push({
      id: state,
      label: getDisplayLabel(state),
      shape: 'ellipse',
      title: showIds ? state : undefined,
      color: { background: bg, border },
      borderWidth: bw,
      ...nodeExtra
    });
  }
  const directorEdgeSet = traceData?.directorEdges ? new Set(traceData.directorEdges) : null;
  function alpha(c, a) {
    if (!c || c.startsWith('rgba')) return c;
    const m = c.match(/^#([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i);
    if (!m) return c;
    return `rgba(${parseInt(m[1],16)},${parseInt(m[2],16)},${parseInt(m[3],16)},${a})`;
  }
  const edgesArr = [];
  let eid = 0;
  for (const [k, t] of edgeMap) {
    const inFront = frontierKeys.has(k);
    const isSel = k === selectedKey;
    const isCtrl = ctrlSet.has(t.action);
    const isDirector = showDirector && directorEdgeSet && directorEdgeSet.has(k);
    const toNode = expanded.has(t.to) ? t.to : UNKNOWN_ID;
    if (toNode === UNKNOWN_ID) {
      hasUnknown = true;
    }
    let color, width, dashes;
    if (isDirector) {
      color = '#15803d';
      width = 3;
      dashes = false;
    } else if (isSel) {
      color = '#000000';
      width = 2.5;
      dashes = false;
    } else if (inFront) {
      color = isCtrl ? '#000000' : '#78350f';
      width = 2;
      dashes = false;
    } else {
      color = isCtrl ? '#000000' : '#78350f';
      width = 1;
      dashes = !hideUnknown;
    }
    if (hideUnknown && toNode === UNKNOWN_ID) {
      color = 'rgba(156, 163, 175, 0.3)';
      width = 1;
      dashes = true;
    }
    if (selectedNodeId !== null && t.from === selectedNodeId) {
      color = '#2563eb';
      width = 2;
      dashes = false;
    }
    let fontColor;
    if (selectedNodeId !== null && t.from === selectedNodeId) {
      fontColor = '#1e40af';
    } else if (isDirector) {
      fontColor = '#15803d';
    } else if (hideUnknown && toNode === UNKNOWN_ID) {
      fontColor = '#6b7280';
    } else if (isSel) {
      fontColor = '#374151';
    } else if (inFront) {
      fontColor = isCtrl ? '#4b5563' : '#92400e';
    } else {
      fontColor = '#c0c0c0';
    }
    if (showDirector && !isDirector) {
      if (hideUnknown && toNode === UNKNOWN_ID) continue;
      color = alpha(color, 0.35);
      fontColor = alpha(fontColor, 0.35);
      width = 0.5;
      dashes = true;
    }
    if (toNode === UNKNOWN_ID && !isDirector && !(selectedNodeId !== null && t.from === selectedNodeId)) dashes = true;
    edgesArr.push({
      id: eid++,
      from: t.from,
      to: toNode,
      label: t.action,
      color: { color, highlight: color, hover: color },
      width,
      dashes,
      font: {
        color: fontColor,
        size: 11,
        background: 'rgba(255,255,255,0.75)'
      }
    });
  }
  if (hasUnknown) {
    const pos = savedPositions[UNKNOWN_ID];
    let ux, uy, ufixed;
    if (pos) {
      ux = pos.x; uy = pos.y; ufixed = { x: true, y: true };
    } else {
      ux = (maxDist + 1) * X_SPACING; uy = 0; ufixed = { x: true, y: false };
    }
    while (usedPos.has(`${ux},${uy}`)) uy += 100;
    usedPos.add(`${ux},${uy}`);
    const nodeExtra = { x: ux, y: uy, fixed: ufixed };
    nodesArr.push({
      id: UNKNOWN_ID,
      label: 'unknown',
      shape: 'box',
      color: { background: '#f3f4f6', border: '#9ca3af' },
      font: { size: 20, color: '#6b7280', bold: true },
      borderWidth: 2,
      shapeProperties: { borderDashes: [8, 4] },
      ...nodeExtra
    });
  }
  return { nodes: nodesArr, edges: edgesArr };
}

function unfixNodes() {
  const pos = traceNetwork.getPositions();
  Object.assign(savedPositions, pos);
  if (currentNodesDS) {
    currentNodesDS.update(
      Object.keys(pos).map(id => ({ id, fixed: { x: false, y: false } }))
    );
  }
  traceNetwork.setOptions({ physics: { enabled: false } });
}

function renderTraceAtStep(step) {
  const { nodes, edges } = buildGraphAtStep(step);
  const hasNew = nodes.some(n => !savedPositions[n.id]);
  currentNodesDS = new vis.DataSet(nodes);
  const edgesDS = new vis.DataSet(spreadParallelEdges([...edges]));
  if (traceNetwork === null) {
    const opts = JSON.parse(JSON.stringify(traceNetOpts));
    opts.physics.enabled = hasNew;
    traceNetwork = new vis.Network(
      document.getElementById('traceNetworkDiv'),
      { nodes: currentNodesDS, edges: edgesDS },
      opts
    );
    traceNetwork.on('click', (params) => {
      if (params.nodes.length > 0) {
        selectedNodeId = params.nodes[0];
        renderTraceAtStep(currentStep);
      }
    });
    traceNetwork.on('dragEnd', (params) => {
      if (params.nodes.length > 0) {
        Object.assign(savedPositions, traceNetwork.getPositions(params.nodes));
      }
    });
  } else {
    const vp = traceNetwork.getViewPosition();
    const vs = traceNetwork.getScale();
    traceNetwork.setOptions({ physics: { enabled: hasNew } });
    traceNetwork.setData({ nodes: currentNodesDS, edges: edgesDS });
    traceNetwork.moveTo({ position: vp, scale: vs });
  }
  if (hasNew) {
    traceNetwork.once('stabilized', unfixNodes);
  } else {
    unfixNodes();
  }
  updateStepDisplay(step);
}

function updateStepDisplay(step) {
  const N = traceData.steps.length;
  const info = document.getElementById('traceStepInfo');
  document.getElementById('stepCounter').textContent = `Step ${step} / ${N}`;
  document.getElementById('stepBackBtn').disabled = step === 0;
  document.getElementById('stepFwdBtn').disabled = step === N;
  document.getElementById('goStartBtn').disabled = step === 0;
  document.getElementById('goEndBtn').disabled = step === N;
  info.innerHTML = '';
  if (step >= N) {
    const msg = document.createElement('div');
    msg.style.width = '100%';
    msg.style.display = 'flex';
    msg.style.flexDirection = 'column';
    msg.style.gap = '6px';
    const title = document.createElement('span');
    title.innerHTML = `<strong>Exploration complete — ${traceData.realizable ? 'REALIZABLE \u2713' : 'UNREALIZABLE \u2717'}</strong>`;
    msg.appendChild(title);
    const dirEdges = traceData.directorEdges || [];
    if (dirEdges.length > 0) {
      const expandedKeys = new Set();
      for (let i = 0; i < N; i++) {
        const t = traceData.steps[i].selected;
        expandedKeys.add(t.from + '\x01' + t.action + '\x01' + t.to);
      }
      const expandedTitle = document.createElement('span');
      expandedTitle.style.marginTop = '4px';
      expandedTitle.innerHTML = '<strong>Director transitions:</strong>';
      msg.appendChild(expandedTitle);
      for (let i = 0; i < dirEdges.length; i++) {
        const parts = dirEdges[i].split('');
        const key = parts[0] + '\x01' + parts[1] + '\x01' + parts[2];
        const wasExpanded = expandedKeys.has(key);
        const row = document.createElement('span');
        row.style.fontWeight = '700';
        row.style.whiteSpace = 'nowrap';
        row.style.fontFamily = 'monospace';
        row.style.color = wasExpanded ? '#000' : '#dc2626';
        row.innerHTML = `<span class="frontier-idx">[${i+1}]</span>${parts[0]} ——[${parts[1]}]——▶ ${parts[2]}`;
        msg.appendChild(row);
      }
    } else {
      const expandedTitle = document.createElement('span');
      expandedTitle.style.marginTop = '4px';
      expandedTitle.innerHTML = '<strong>Expanded transitions:</strong>';
      msg.appendChild(expandedTitle);
      for (let i = 0; i < N; i++) {
        const s = traceData.steps[i];
        const t = s.selected;
        const row = document.createElement('span');
        row.className = 'frontier-item selected';
        row.innerHTML = `<span class="frontier-idx">[${i+1}]</span>${t.from} ——[${t.action}]——▶ ${t.to}`;
        msg.appendChild(row);
      }
    }
    info.appendChild(msg);
    return;
  }
  const s = traceData.steps[step];
  for (let i = 0; i < s.frontier.length; i++) {
    const t = s.frontier[i];
    const sel = i === s.selectedIndex;
    const span = document.createElement('span');
    span.className = 'frontier-item ' + (sel ? 'selected' : 'unselected');
    span.innerHTML = `<span class="frontier-idx">[${i+1}]</span>${t.from} \u2014\u2014[${t.action}]\u2014\u2014\u25b6 ${t.to}`;
    info.appendChild(span);
  }
}

function togglePlay() {
  if (!traceData) return;
  if (playInterval) {
    clearInterval(playInterval);
    playInterval = null;
    document.getElementById('playBtn').textContent = '\u25b6 Play';
    document.getElementById('playBtn').classList.remove('playing');
  } else {
    document.getElementById('playBtn').textContent = '\u23f8 Pause';
    document.getElementById('playBtn').classList.add('playing');
    playInterval = setInterval(() => {
      const N = traceData.steps.length;
      if (currentStep >= N) {
        clearInterval(playInterval);
        playInterval = null;
        document.getElementById('playBtn').textContent = '\u25b6 Play';
        document.getElementById('playBtn').classList.remove('playing');
        return;
      }
      currentStep++;
      selectedNodeId = null;
      renderTraceAtStep(currentStep);
    }, 1000);
  }
}

function initTraceTabIfNeeded() {
  if (traceInited) return;
  traceInited = true;
  if (!traceData) {
    document.getElementById('traceNoData').style.display = '';
    return;
  }
  document.getElementById('traceNoData').style.display = 'none';
  document.getElementById('traceContent').style.display = '';
  const meta = document.getElementById('traceMeta');
  const dirEdges = traceData.directorEdges || [];
  [
    ['Instance', traceData.instanceName],
    ['Mode', traceData.mode],
    ['Heuristic', traceData.heuristic],
    ['Steps', traceData.steps.length],
    ['Director', dirEdges.length],
    ['File', traceData.solFile]
  ].forEach(([k, v]) => {
    const span = document.createElement('span');
    span.className = 'trace-badge';
    span.innerHTML = `<strong>${k}:</strong> ${v}`;
    meta.appendChild(span);
  });
  renderTraceAtStep(0);
}

document.getElementById('stepBackBtn').addEventListener('click', () => {
  if (!traceData || currentStep === 0) return;
  currentStep--;
  selectedNodeId = null;
  renderTraceAtStep(currentStep);
});

document.getElementById('stepFwdBtn').addEventListener('click', () => {
  if (!traceData || currentStep >= traceData.steps.length) return;
  currentStep++;
  selectedNodeId = null;
  renderTraceAtStep(currentStep);
});

document.getElementById('goStartBtn').addEventListener('click', () => {
  if (!traceData) return;
  currentStep = 0;
  selectedNodeId = null;
  renderTraceAtStep(currentStep);
});

document.getElementById('goEndBtn').addEventListener('click', () => {
  if (!traceData) return;
  currentStep = traceData.steps.length;
  selectedNodeId = null;
  renderTraceAtStep(currentStep);
});

document.getElementById('playBtn').addEventListener('click', togglePlay);

document.getElementById('resetZoomBtn').addEventListener('click', () => {
  if (traceNetwork) traceNetwork.fit();
});

document.getElementById('directorBtn').addEventListener('click', toggleDirector);
