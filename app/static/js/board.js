/**
 * app/static/js/board.js - WardCast Wall Display Board Controller
 * Auto-updates the wall TV display from /api/surge_state and /api/forecast_7day.
 */

async function updateBoard() {
  try {
    const res = await fetch('/api/surge_state').then(r => r.json());
    renderBoard(res);
  } catch (err) {
    console.error('Failed to update board:', err);
  }
}

function renderBoard(surge) {
  const ruleLevel = surge.rule_warning_level || 'NORMAL';
  const occRate = surge.latest_occupancy_rate || 0.92;
  const unmetDemand = surge.unmet_demand_beds || 0;
  const capacity = 120;
  const inUse = Math.min(capacity, Math.round(occRate * capacity));
  const free = Math.max(0, capacity - inUse);

  // Status Badge
  const badge = document.getElementById('board-status-badge');
  const badgeText = document.getElementById('board-status-text');
  const actionText = document.getElementById('board-action-text');
  const bar = document.getElementById('board-capacity-bar');
  const capLabel = document.getElementById('board-capacity-label');
  const bedBlock = document.getElementById('board-bed-block');

  const occPct = Math.round(occRate * 100);

  if (ruleLevel === 'CRITICAL SURGE' || inUse >= 108) {
    document.body.className = 'w-screen h-screen flex flex-col justify-between p-16 select-none bg-gradient-to-b from-[#fff5f5] via-[#fee2e2]/60 to-[#fecaca]/50 text-[#7f1d1d]';
    badge.className = 'inline-flex items-center gap-6 bg-[#dc2626] text-white px-12 py-5 rounded-3xl shadow-xl shadow-red-600/30 mb-8';
    badgeText.textContent = 'CRITICAL';
    actionText.textContent = 'Pause non-urgent admissions. Speed up discharges. Open reserve overflow beds.';
    bar.className = 'h-full bg-gradient-to-r from-red-500 to-[#dc2626] rounded-full';
  } else if (ruleLevel === 'SURGE ALERT (PRE-SURGE)' || inUse >= 102) {
    document.body.className = 'w-screen h-screen flex flex-col justify-between p-16 select-none bg-gradient-to-b from-[#fffbeb] via-[#fef3c7]/60 to-[#fde68a]/50 text-[#92400e]';
    badge.className = 'inline-flex items-center gap-6 bg-[#d97706] text-white px-12 py-5 rounded-3xl shadow-xl shadow-amber-600/30 mb-8';
    badgeText.textContent = 'SURGE ALERT';
    actionText.textContent = 'Notify on-call nursing teams. Prepare discharge lounge. Expedite morning rounds.';
    bar.className = 'h-full bg-gradient-to-r from-amber-500 to-[#d97706] rounded-full';
  } else {
    document.body.className = 'w-screen h-screen flex flex-col justify-between p-16 select-none bg-gradient-to-b from-[#f0fdf4] via-[#dcfce7]/60 to-[#bbf7d0]/50 text-[#14532d]';
    badge.className = 'inline-flex items-center gap-6 bg-[#16a34a] text-white px-12 py-5 rounded-3xl shadow-xl shadow-green-600/30 mb-8';
    badgeText.textContent = 'NORMAL';
    actionText.textContent = 'Standard patient intake and routine discharge rounds.';
    bar.className = 'h-full bg-gradient-to-r from-green-500 to-[#16a34a] rounded-full';
  }

  // Bed Numbers
  let bedHtml = `<span class="text-[#991b1b] font-black">${inUse}</span> of <span class="font-bold">120</span> beds in use <span class="mx-3 text-[#94a3b8] font-light">·</span> <span class="text-[#dc2626] font-black">${free}</span> free`;
  if (unmetDemand > 0) {
    bedHtml += ` <span class="mx-3 text-[#94a3b8] font-light">·</span> <span class="text-[#dc2626] font-black">${unmetDemand}</span> waiting with no bed`;
  }
  bedBlock.innerHTML = bedHtml;

  // Capacity visual scale bar (never exceeds 100%)
  const widthPct = Math.min(100.0, (inUse / 120.0) * 100.0);
  bar.style.width = widthPct + '%';
  capLabel.textContent = `${occPct}% Capacity Occupied`;

  // Time updater
  const now = new Date();
  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');
  document.getElementById('board-time').textContent = `Updated ${hours}:${minutes}`;
}

// Initial update and auto-poll every 15 seconds
document.addEventListener('DOMContentLoaded', () => {
  updateBoard();
  setInterval(updateBoard, 15000);
});
