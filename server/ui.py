from __future__ import annotations


def render_dashboard_html() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>OpsEnv Governance Dashboard</title>
  <style>
    body { margin: 0; font-family: Inter, Arial, sans-serif; background: #0b1020; color: #e6edf7; }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 20px; }
    h1 { margin: 0 0 8px; font-size: 24px; }
    .sub { color: #9fb0d1; margin-bottom: 18px; }
    .grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(240px,1fr)); }
    .card { background: #121a30; border: 1px solid #27314e; border-radius: 10px; padding: 14px; }
    .label { color: #9fb0d1; font-size: 12px; }
    .value { font-size: 24px; font-weight: 700; margin-top: 6px; }
    .ok { color: #4ade80; } .warn { color: #facc15; } .bad { color: #f87171; }
    table { width: 100%; border-collapse: collapse; margin-top: 14px; }
    th, td { border-bottom: 1px solid #27314e; text-align: left; padding: 8px; font-size: 13px; }
    .footer { margin-top: 12px; color: #9fb0d1; font-size: 12px; }
    button { background:#2563eb;color:#fff;border:none;padding:8px 12px;border-radius:6px;cursor:pointer; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>OpsEnv Governance Dashboard</h1>
    <div class="sub">Live environment health, learning, adversarial robustness, and trust score</div>
    <div style="margin-bottom:10px;"><button onclick="refresh()">Refresh</button></div>
    <div class="grid">
      <div class="card"><div class="label">Service Status</div><div id="status" class="value">-</div></div>
      <div class="card"><div class="label">Production Trust Score</div><div id="trust" class="value">-</div></div>
      <div class="card"><div class="label">Kaizen Episodes</div><div id="episodes" class="value">-</div></div>
      <div class="card"><div class="label">Recent Avg Reward</div><div id="reward" class="value">-</div></div>
      <div class="card"><div class="label">Adversarial Solver Failure Rate</div><div id="solverFail" class="value">-</div></div>
      <div class="card"><div class="label">Counterfactual Avg Regret</div><div id="regret" class="value">-</div></div>
    </div>
    <div class="card" style="margin-top:12px;">
      <div class="label">Live Trends (Trust Score & Reward)</div>
      <canvas id="trendChart" width="1000" height="180" style="width:100%;height:180px;background:#0f1730;border-radius:8px;"></canvas>
      <div class="footer">Source: `/metrics/live` (rolling snapshots)</div>
    </div>
    <div class="card" style="margin-top:12px;">
      <div class="label">Trust Inputs</div>
      <table>
        <thead><tr><th>Signal</th><th>Value</th></tr></thead>
        <tbody id="trustRows"></tbody>
      </table>
      <div class="footer">Data source: `/metrics`, `/trust/score`</div>
    </div>
  </div>
<script>
function fmtPct(v){ return (v*100).toFixed(1) + "%"; }
function tone(num){ if(num>=0.8) return "ok"; if(num>=0.6) return "warn"; return "bad"; }
function drawTrends(history){
  const c = document.getElementById('trendChart');
  const ctx = c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  if(!history || history.length < 2){ return; }
  const trust = history.map(x => Number(x.trust_score ?? 0));
  const reward = history.map(x => Number(x.avg_reward_recent ?? 0));
  const maxTrust = Math.max(...trust, 1);
  const maxReward = Math.max(...reward, 1);
  const minReward = Math.min(...reward, -1);
  const pad = 20;
  const w = c.width - pad*2;
  const h = c.height - pad*2;
  const step = w / Math.max(1, history.length - 1);

  // Trust line
  ctx.beginPath();
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2;
  trust.forEach((v,i)=>{
    const x = pad + i*step;
    const y = pad + h - (v / maxTrust) * h;
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.stroke();

  // Reward line
  ctx.beginPath();
  ctx.strokeStyle = '#22c55e';
  ctx.lineWidth = 2;
  reward.forEach((v,i)=>{
    const x = pad + i*step;
    const y = pad + h - ((v - minReward) / Math.max(0.001, (maxReward - minReward))) * h;
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.stroke();

  ctx.fillStyle = '#9fb0d1';
  ctx.font = '12px Inter';
  ctx.fillText('Trust Score', 12, 14);
  ctx.fillStyle = '#22c55e';
  ctx.fillText('Avg Reward', 100, 14);
}
async function refresh(){
  const [mRes, tRes, lRes] = await Promise.all([fetch('/metrics'), fetch('/trust/score'), fetch('/metrics/live?limit=60')]);
  const m = await mRes.json();
  const t = await tRes.json();
  const live = await lRes.json();
  const status = m.health?.status || "unknown";
  const statusEl = document.getElementById('status');
  statusEl.textContent = status.toUpperCase();
  statusEl.className = "value " + (status === "healthy" ? "ok" : "bad");
  document.getElementById('trust').textContent = `${t.score} (${t.grade})`;
  document.getElementById('trust').className = "value " + tone((t.score||0)/100);
  document.getElementById('episodes').textContent = m.kaizen?.episode_count ?? 0;
  document.getElementById('reward').textContent = Number(m.kaizen?.avg_reward_recent ?? 0).toFixed(4);
  document.getElementById('solverFail').textContent = fmtPct(Number(m.adversarial?.solver_failure_rate ?? 0));
  const regretVal = Number(m.counterfactual?.avg_regret ?? 0);
  document.getElementById('regret').textContent = regretVal.toFixed(4);
  document.getElementById('regret').className = "value " + (regretVal < 0.2 ? "ok" : regretVal < 0.4 ? "warn" : "bad");
  const rows = Object.entries(t.inputs || {}).map(([k,v]) => `<tr><td>${k}</td><td>${Number(v).toFixed(4)}</td></tr>`).join("");
  document.getElementById('trustRows').innerHTML = rows;
  drawTrends(live.history || []);
}
refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>
""".strip()
