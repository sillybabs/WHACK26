function renderBalance(score) {
  const pct = (score.balance * 100).toFixed(0);
  document.getElementById('bal-green').style.height = pct + '%';
  document.getElementById('bal-pct').textContent = pct + '%';   // number beside the bar for colourblind safety
}