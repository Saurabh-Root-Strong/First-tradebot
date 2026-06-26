/* TradingView-style crosshair value tags for the Plotly charts.
 *
 * Plotly's spikelines draw the dotted vertical/horizontal guides, but it has no
 * native "value at the cursor" axis label. This attaches one: as the mouse moves
 * over a chart it converts the cursor pixel back to data coordinates via the
 * Plotly axis objects and floats two tags —
 *   • a PRICE tag pinned to the left y-axis of WHICHEVER stacked panel the cursor
 *     is in (price / OI / premium / flow each read their own y-axis), and
 *   • a TIME tag pinned to the bottom x-axis.
 * Pure presentation; reads gd._fullLayout only, never mutates the figure.
 *
 * Dash auto-serves everything in assets/, so this loads on every page with no
 * Python wiring. A MutationObserver re-attaches as graphs (incl. modal popups)
 * mount/redraw.
 */
(function () {
  "use strict";

  var ACCENT = "#7dd3fc";   // matches the spikeline colour
  var INK    = "#0a0f1a";

  function mkTag() {
    var t = document.createElement("div");
    t.style.cssText =
      "position:absolute;z-index:1000;pointer-events:none;display:none;" +
      "background:" + ACCENT + ";color:" + INK + ";" +
      "font:700 10px ui-monospace,Menlo,Consolas,monospace;" +
      "padding:1px 4px;border-radius:2px;white-space:nowrap;";
    return t;
  }

  // The non-overlay y-axis whose pixel band contains cursor y (skip secondary
  // axes — volume / IV — that overlay the same band as their primary).
  function yAxisAt(fl, y) {
    for (var k in fl) {
      if (k.indexOf("yaxis") !== 0) continue;
      var ya = fl[k];
      if (!ya || !ya._length || ya.overlaying) continue;
      if (y >= ya._offset && y <= ya._offset + ya._length) return ya;
    }
    return null;
  }

  // Pixel -> LINEAR axis value, derived straight from range + pixel band so the
  // result can't drift from the visible spike (Plotly's p2d/p2c handle _offset
  // inconsistently across versions — that mismatch was the ~130-pt error).
  //   vertical=true  (y-axis): top pixel = range[1] (high), bottom = range[0]
  //   vertical=false (x-axis): left pixel = range[0],        right  = range[1]
  // r2l linearises the range (identity for a linear price axis; epoch-ms for a
  // date axis). Caller maps back: prices use the value as-is, time via Date(ms).
  function p2lin(ax, px, vertical) {
    if (!ax || !ax._length) return null;
    var lo = ax.r2l(ax.range[0]), hi = ax.r2l(ax.range[1]);
    var f = (px - ax._offset) / ax._length;
    return vertical ? hi - f * (hi - lo) : lo + f * (hi - lo);
  }

  function fmtPrice(v) {
    if (v == null || !isFinite(v)) return "";
    var abs = Math.abs(v);
    var dp = abs >= 1000 ? 0 : abs >= 10 ? 1 : 2;
    return Number(v).toLocaleString(undefined,
      { minimumFractionDigits: dp, maximumFractionDigits: dp });
  }

  function fmtTime(v) {
    var d = new Date(v);
    if (isNaN(d)) return "";
    // Plotly stores the chart's naive IST wall-clock timestamps as UTC epoch-ms
    // (and renders its own axis ticks in UTC), so read them back in UTC. Using
    // toLocaleTimeString re-applies the browser's LOCAL tz, which on an IST
    // machine shifted an 11:30 bar to "04:07 pm" (+05:30) — the long-standing bug.
    var hh = String(d.getUTCHours()).padStart(2, "0");
    var mm = String(d.getUTCMinutes()).padStart(2, "0");
    return hh + ":" + mm;
  }

  function attach(gd) {
    if (!gd || gd._cvCrosshair) return;
    gd._cvCrosshair = true;
    gd.style.position = "relative";

    var pTag = mkTag(), tTag = mkTag();
    gd.appendChild(pTag);
    gd.appendChild(tTag);

    function hide() { pTag.style.display = tTag.style.display = "none"; }

    gd.addEventListener("mouseleave", hide);
    gd.addEventListener("mousemove", function (e) {
      var fl = gd._fullLayout, sz = fl && fl._size;
      if (!sz) { hide(); return; }
      var r = gd.getBoundingClientRect();
      var x = e.clientX - r.left, y = e.clientY - r.top;

      // inside the plotting region only
      if (x < sz.l || x > sz.l + sz.w || y < sz.t || y > sz.t + sz.h) { hide(); return; }

      // PRICE tag — y-axis of the panel under the cursor, pinned to its left edge.
      var ya = yAxisAt(fl, y);
      if (ya) {
        pTag.textContent = fmtPrice(p2lin(ya, y, true));
        pTag.style.top = y + "px";
        pTag.style.left = sz.l + "px";
        pTag.style.transform = "translate(-100%,-50%)";
        pTag.style.display = "block";
      } else { pTag.style.display = "none"; }

      // TIME tag — shared x-axis, pinned to the bottom baseline.
      var xa = fl.xaxis;
      if (xa && xa._length) {
        tTag.textContent = fmtTime(p2lin(xa, x, false));   // epoch-ms -> Date
        tTag.style.left = x + "px";
        tTag.style.top = (sz.t + sz.h) + "px";
        tTag.style.transform = "translate(-50%,0)";
        tTag.style.display = "block";
      } else { tTag.style.display = "none"; }
    });
  }

  function scan() {
    document.querySelectorAll(".js-plotly-plot").forEach(attach);
  }

  if (document.readyState !== "loading") scan();
  document.addEventListener("DOMContentLoaded", scan);
  new MutationObserver(scan).observe(document.documentElement, {
    childList: true, subtree: true,
  });
})();
