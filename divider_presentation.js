const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "32-bit Signed Fixed-Point Divider";
pres.author = "K Francis Angel Manohar & Manjunath";

// ─── LIGHT, CLEAN, PROFESSIONAL PALETTE ────────────────────────
const C = {
  bg:        "FFFFFF",   // page background
  surface:   "F8FAFC",   // soft surface
  card:      "FFFFFF",   // card background
  border:    "E5E7EB",   // hairline borders
  hairline:  "EEF2F6",   // subtle dividers
  text:      "1F2937",   // primary text (slate-800)
  subtext:   "6B7280",   // secondary text (slate-500)
  muted:     "9CA3AF",   // tertiary text
  primary:   "2563EB",   // calm blue (blue-600)
  primaryL:  "DBEAFE",   // light blue tint
  accent:    "0EA5E9",   // cyan/sky-500 accent
  teal:      "14B8A6",   // teal-500
  tealL:     "CCFBF1",
  amber:     "F59E0B",   // amber-500
  amberL:    "FEF3C7",
  violet:    "8B5CF6",   // violet-500
  violetL:   "EDE9FE",
  rose:      "F43F5E",
  roseL:     "FFE4E6",
  green:     "10B981",
  greenL:    "D1FAE5",
};

const ASSET = "/tmp/ppt_assets";

// ─── HELPERS ─────────────────────────────────────────────────────
function pageBackground(slide) {
  slide.background = { color: C.bg };
}

function topAccentBar(slide, color) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.05,
    fill: { color: color || C.primary }, line: { color: color || C.primary }
  });
}

function sectionTitle(slide, eyebrow, title) {
  pageBackground(slide);
  topAccentBar(slide);
  if (eyebrow) {
    slide.addText(eyebrow.toUpperCase(), {
      x: 0.5, y: 0.18, w: 9, h: 0.3,
      fontSize: 10, fontFace: "Helvetica", bold: true,
      color: C.primary, charSpacing: 4,
    });
  }
  slide.addText(title, {
    x: 0.5, y: 0.42, w: 9, h: 0.55,
    fontSize: 22, fontFace: "Helvetica", bold: true,
    color: C.text,
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.0, w: 9, h: 0.02,
    fill: { color: C.hairline }, line: { color: C.hairline }
  });
  // Page footer
  slide.addText("32-bit Signed Fixed-Point Divider  ·  IIIT Senapati", {
    x: 0.5, y: 5.32, w: 6, h: 0.25,
    fontSize: 9, fontFace: "Helvetica", color: C.muted,
  });
}

function card(slide, x, y, w, h, opts) {
  opts = opts || {};
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: opts.fill || C.card },
    line: { color: opts.border || C.border, width: 0.5 },
  });
}

function bullets(slide, items, x, y, w, h, opts) {
  opts = opts || {};
  const rows = items.map((item, i) => ({
    text: item,
    options: {
      bullet: { code: "25CF" },
      breakLine: i < items.length - 1,
      fontSize: opts.fontSize || 12,
      color: opts.color || C.text,
      fontFace: "Helvetica",
    }
  }));
  slide.addText(rows, { x, y, w, h, valign: "top", paraSpaceAfter: 4 });
}

function pill(slide, x, y, w, h, label, color) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: color }, line: { color: color }
  });
  slide.addText(label, {
    x, y, w, h,
    fontSize: 9, bold: true, color: C.bg,
    align: "center", valign: "middle", fontFace: "Helvetica",
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 1 — TITLE
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  // Decorative geometric panel on the left
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 3.6, h: 5.625,
    fill: { color: C.surface }, line: { color: C.surface }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.08, h: 5.625,
    fill: { color: C.primary }, line: { color: C.primary }
  });
  // Soft accent dots
  s.addShape(pres.shapes.OVAL, { x: 0.4, y: 0.5, w: 0.18, h: 0.18, fill: { color: C.primary }, line: { color: C.primary } });
  s.addShape(pres.shapes.OVAL, { x: 0.7, y: 0.5, w: 0.18, h: 0.18, fill: { color: C.teal }, line: { color: C.teal } });
  s.addShape(pres.shapes.OVAL, { x: 1.0, y: 0.5, w: 0.18, h: 0.18, fill: { color: C.amber }, line: { color: C.amber } });

  s.addText("MINOR PROJECT  ·  EC3201  ·  APRIL 2026", {
    x: 0.4, y: 1.0, w: 3.0, h: 0.3,
    fontSize: 9, color: C.subtext, charSpacing: 3, bold: true, fontFace: "Helvetica",
  });
  s.addText("FPGA & ASIC", {
    x: 0.4, y: 1.4, w: 3.0, h: 0.4,
    fontSize: 14, color: C.primary, bold: true, fontFace: "Helvetica",
  });
  s.addText("32-bit Signed\nFixed-Point\nDivider", {
    x: 0.4, y: 1.85, w: 3.0, h: 2.2,
    fontSize: 30, color: C.text, bold: true, fontFace: "Helvetica",
    paraSpaceAfter: 2,
  });
  s.addText("Comparative Hardware Study of\nRestoring · Newton-Raphson · Goldschmidt", {
    x: 0.4, y: 4.1, w: 3.0, h: 0.7,
    fontSize: 11, color: C.subtext, italic: true, fontFace: "Helvetica",
  });

  // Right content — authors
  s.addText("Authors", {
    x: 4.3, y: 2.0, w: 5, h: 0.3,
    fontSize: 10, color: C.subtext, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  s.addText("K Francis Angel Manohar  ·  230104004\nManjunath  ·  230102073", {
    x: 4.3, y: 2.3, w: 5, h: 0.7,
    fontSize: 14, color: C.text, fontFace: "Helvetica",
  });
  s.addText("Supervisor", {
    x: 4.3, y: 3.2, w: 5, h: 0.3,
    fontSize: 10, color: C.subtext, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  s.addText("Dr. Dilip Singh\nAssistant Professor, Dept. of ECE", {
    x: 4.3, y: 3.5, w: 5, h: 0.7,
    fontSize: 13, color: C.text, fontFace: "Helvetica",
  });
  s.addText("Indian Institute of Information Technology, Senapati, Manipur", {
    x: 4.3, y: 4.4, w: 5, h: 0.3,
    fontSize: 10, color: C.muted, fontFace: "Helvetica",
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 2 — OUTLINE
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Agenda", "Presentation Outline");

  const items = [
    ["01", "Motivation & Problem Statement", C.primary],
    ["02", "Division Algorithms — Overview", C.teal],
    ["03", "Architecture Design (Flowcharts & FSMs)", C.violet],
    ["04", "ASIC Synthesis Results", C.amber],
    ["05", "RTL Simulation & Verification", C.primary],
    ["06", "FPGA Standalone IP Core", C.teal],
    ["07", "VIO-Integrated Hardware Verification", C.violet],
    ["08", "Physical Hardware Demonstration", C.amber],
    ["09", "Comparison, Conclusion & Future Work", C.primary],
  ];

  items.forEach(([num, text, color], i) => {
    const col = i < 5 ? 0 : 1;
    const row = i < 5 ? i : i - 5;
    const x = col === 0 ? 0.5 : 5.05;
    const y = 1.2 + row * 0.78;

    card(s, x, y, 4.45, 0.65);
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.06, h: 0.65, fill: { color }, line: { color } });
    s.addText(num, {
      x: x + 0.18, y, w: 0.5, h: 0.65,
      fontSize: 14, bold: true, color, align: "left", valign: "middle", fontFace: "Helvetica",
    });
    s.addText(text, {
      x: x + 0.7, y, w: 3.7, h: 0.65,
      fontSize: 12, color: C.text, valign: "middle", fontFace: "Helvetica",
    });
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 3 — MOTIVATION
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Chapter 1", "Motivation & Problem Statement");

  // Left card: Motivation
  card(s, 0.5, 1.2, 4.4, 3.8, { fill: C.surface });
  s.addText("Why this matters", {
    x: 0.7, y: 1.32, w: 4.0, h: 0.3,
    fontSize: 10, color: C.primary, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  s.addText("Division — the slowest arithmetic unit", {
    x: 0.7, y: 1.6, w: 4.0, h: 0.4,
    fontSize: 15, bold: true, color: C.text, fontFace: "Helvetica",
  });
  bullets(s, [
    "Most complex of the four basic operations",
    "Drives throughput in DSP, AI (BatchNorm, Softmax), and edge computing",
    "Balances Iron Triangle: Speed · Area · Power",
    "Critical path for next-gen ASIC and FPGA accelerators",
    "Existing literature focuses on 64-bit FPUs — gap in 32-bit fixed-point",
  ], 0.7, 2.05, 4.0, 2.85, { fontSize: 12 });

  // Right card: Three deficits
  card(s, 5.1, 1.2, 4.4, 3.8);
  s.addText("Three core deficits", {
    x: 5.3, y: 1.32, w: 4.0, h: 0.3,
    fontSize: 10, color: C.primary, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });

  const deficits = [
    { t: "Area–Latency Conflict", d: "High-speed algorithms need large multipliers", c: C.primary },
    { t: "Signed Operand Complexity", d: "Two's complement adds sign-handling logic", c: C.teal },
    { t: "Precision & Convergence Errors", d: "Iterative methods need careful bit-width control", c: C.violet },
  ];
  deficits.forEach((d, i) => {
    const y = 1.7 + i * 1.05;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.3, y, w: 0.05, h: 0.85, fill: { color: d.c }, line: { color: d.c } });
    s.addText(d.t, { x: 5.45, y: y + 0.02, w: 4.0, h: 0.34, fontSize: 13, bold: true, color: C.text, fontFace: "Helvetica" });
    s.addText(d.d, { x: 5.45, y: y + 0.38, w: 4.0, h: 0.45, fontSize: 11, color: C.subtext, fontFace: "Helvetica" });
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 4 — ALGORITHM OVERVIEW
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Chapter 2", "Division Algorithm Classification");

  const algos = [
    {
      name: "Restoring",
      sub: "Digit-Recurrence",
      color: C.primary, lcolor: C.primaryL,
      points: [
        "Shift & subtract — one bit per cycle",
        "32 cycles for 32-bit operand",
        "Linear convergence — O(n)",
        "Adder + shift registers only",
      ],
      tag: "Area-Efficient",
    },
    {
      name: "Newton-Raphson",
      sub: "Iterative Multiplicative",
      color: C.teal, lcolor: C.tealL,
      points: [
        "x\u208A\u2081 = x(2 − Dx)",
        "Quadratic — O(log n)",
        "4–6 cycles for 32-bit",
        "Multiplier + 256-word LUT seed",
      ],
      tag: "Balanced",
    },
    {
      name: "Goldschmidt",
      sub: "Iterative Multiplicative",
      color: C.violet, lcolor: C.violetL,
      points: [
        "Parallel N & D updates: Nₖ₊₁ = Nₖ·Fₖ",
        "Quadratic — O(log n)",
        "3–5 cycles for 32-bit",
        "Dual multiplier paths + LUT",
      ],
      tag: "High-Speed",
    },
  ];

  algos.forEach((a, i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 1.2, 2.95, 3.95);
    // Header strip
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.2, w: 2.95, h: 0.5, fill: { color: a.lcolor }, line: { color: a.lcolor } });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.2, w: 0.05, h: 0.5, fill: { color: a.color }, line: { color: a.color } });
    s.addText(a.name, { x: x + 0.15, y: 1.22, w: 2.7, h: 0.28, fontSize: 14, bold: true, color: a.color, fontFace: "Helvetica" });
    s.addText(a.sub, { x: x + 0.15, y: 1.46, w: 2.7, h: 0.22, fontSize: 9, italic: true, color: C.subtext, fontFace: "Helvetica" });

    a.points.forEach((p, j) => {
      const py = 1.85 + j * 0.55;
      s.addShape(pres.shapes.OVAL, { x: x + 0.18, y: py + 0.1, w: 0.1, h: 0.1, fill: { color: a.color }, line: { color: a.color } });
      s.addText(p, { x: x + 0.35, y: py, w: 2.5, h: 0.5, fontSize: 11, color: C.text, valign: "top", fontFace: "Helvetica" });
    });

    pill(s, x + 0.5, 4.65, 1.95, 0.32, a.tag, a.color);
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 5 — RESTORING FLOWCHART & FSM
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Chapter 3.2", "Restoring Division — Flowchart & FSM");

  // Left: Flowchart image
  card(s, 0.5, 1.2, 4.4, 3.95);
  s.addText("Algorithmic Flowchart  ·  Figure 3.1", {
    x: 0.65, y: 1.28, w: 4.1, h: 0.28, fontSize: 10, color: C.primary, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  s.addImage({ path: `${ASSET}/restoring_flowchart.png`, x: 0.85, y: 1.6, w: 3.7, h: 3.45, sizing: { type: "contain", w: 3.7, h: 3.45 } });

  // Right: FSM image + commentary
  card(s, 5.1, 1.2, 4.4, 3.95);
  s.addText("FSM State Diagram  ·  Figure 3.2", {
    x: 5.25, y: 1.28, w: 4.1, h: 0.28, fontSize: 10, color: C.primary, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  s.addImage({ path: `${ASSET}/restoring_fsm.png`, x: 5.3, y: 1.6, w: 4.0, h: 2.55, sizing: { type: "contain", w: 4.0, h: 2.55 } });

  // Key facts strip
  card(s, 5.25, 4.3, 4.1, 0.75, { fill: C.surface });
  const facts = [["States", "7"], ["Cycles", "32"], ["Convergence", "Linear"], ["Hardware", "Subtractor + Shift"]];
  facts.forEach(([k, v], i) => {
    const fx = 5.35 + i * 1.0;
    s.addText(k, { x: fx, y: 4.35, w: 1.0, h: 0.22, fontSize: 8, color: C.muted, bold: true, charSpacing: 2, fontFace: "Helvetica", align: "center" });
    s.addText(v, { x: fx, y: 4.55, w: 1.0, h: 0.42, fontSize: 12, bold: true, color: C.primary, fontFace: "Helvetica", align: "center" });
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 6 — NEWTON-RAPHSON FLOWCHART & FSM
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Chapter 3.3", "Newton-Raphson — Flowchart & FSM");

  card(s, 0.5, 1.2, 4.4, 3.95);
  s.addText("Iterative Reciprocal Flowchart", {
    x: 0.65, y: 1.28, w: 4.1, h: 0.28, fontSize: 10, color: C.teal, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  s.addImage({ path: `${ASSET}/nr_flowchart.png`, x: 0.85, y: 1.6, w: 3.7, h: 3.45, sizing: { type: "contain", w: 3.7, h: 3.45 } });

  card(s, 5.1, 1.2, 4.4, 3.95);
  s.addText("FSM State Diagram  ·  Figure 3.4", {
    x: 5.25, y: 1.28, w: 4.1, h: 0.28, fontSize: 10, color: C.teal, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  s.addImage({ path: `${ASSET}/nr_fsm.png`, x: 5.2, y: 1.6, w: 4.2, h: 2.55, sizing: { type: "contain", w: 4.2, h: 2.55 } });

  card(s, 5.25, 4.3, 4.1, 0.75, { fill: C.surface });
  const facts = [["States", "6"], ["Cycles", "4–6"], ["Convergence", "Quadratic"], ["Hardware", "Multiplier + LUT"]];
  facts.forEach(([k, v], i) => {
    const fx = 5.35 + i * 1.0;
    s.addText(k, { x: fx, y: 4.35, w: 1.0, h: 0.22, fontSize: 8, color: C.muted, bold: true, charSpacing: 2, fontFace: "Helvetica", align: "center" });
    s.addText(v, { x: fx, y: 4.55, w: 1.0, h: 0.42, fontSize: 12, bold: true, color: C.teal, fontFace: "Helvetica", align: "center" });
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 7 — GOLDSCHMIDT FLOWCHART & FSM
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Chapter 3.4", "Goldschmidt — Flowchart & FSM");

  card(s, 0.5, 1.2, 4.4, 3.95);
  s.addText("Parallel Convergence Flowchart  ·  Figure 3.5", {
    x: 0.65, y: 1.28, w: 4.1, h: 0.28, fontSize: 10, color: C.violet, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  s.addImage({ path: `${ASSET}/gold_flowchart.png`, x: 0.85, y: 1.6, w: 3.7, h: 3.45, sizing: { type: "contain", w: 3.7, h: 3.45 } });

  card(s, 5.1, 1.2, 4.4, 3.95);
  s.addText("FSM State Diagram  ·  Figure 3.6", {
    x: 5.25, y: 1.28, w: 4.1, h: 0.28, fontSize: 10, color: C.violet, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  s.addImage({ path: `${ASSET}/gold_fsm.png`, x: 5.5, y: 1.6, w: 3.6, h: 2.55, sizing: { type: "contain", w: 3.6, h: 2.55 } });

  card(s, 5.25, 4.3, 4.1, 0.75, { fill: C.surface });
  const facts = [["States", "5"], ["Cycles", "3–5"], ["Convergence", "Quadratic"], ["Hardware", "Parallel Mults"]];
  facts.forEach(([k, v], i) => {
    const fx = 5.35 + i * 1.0;
    s.addText(k, { x: fx, y: 4.35, w: 1.0, h: 0.22, fontSize: 8, color: C.muted, bold: true, charSpacing: 2, fontFace: "Helvetica", align: "center" });
    s.addText(v, { x: fx, y: 4.55, w: 1.0, h: 0.42, fontSize: 12, bold: true, color: C.violet, fontFace: "Helvetica", align: "center" });
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 8 — SIGNED LOGIC & FSM TABLE
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Chapter 3.1", "Signed Wrapper & FSM Comparison");

  // Signed wrapper
  card(s, 0.5, 1.2, 4.4, 1.85, { fill: C.surface });
  s.addText("Signed Wrapper Logic", { x: 0.65, y: 1.28, w: 4.1, h: 0.28, fontSize: 10, color: C.primary, bold: true, charSpacing: 2, fontFace: "Helvetica" });
  bullets(s, [
    "Absolute magnitude extraction (|D|, |V|)",
    "Sign XOR — Signout = SignD ⊕ SignV",
    "Two's complement correction on output",
    "Core engines operate purely on unsigned values",
  ], 0.65, 1.55, 4.1, 1.45, { fontSize: 11.5 });

  // Format card
  card(s, 5.1, 1.2, 4.4, 1.85);
  s.addText("Q16.16 Fixed-Point Format", { x: 5.25, y: 1.28, w: 4.1, h: 0.28, fontSize: 10, color: C.teal, bold: true, charSpacing: 2, fontFace: "Helvetica" });
  s.addText("16 integer bits  ·  16 fractional bits", { x: 5.25, y: 1.55, w: 4.1, h: 0.3, fontSize: 13, bold: true, color: C.text, fontFace: "Helvetica" });
  bullets(s, [
    "32-bit signed two's complement",
    "Range: [-32768.0, +32767.99998]",
    "Resolution: 1/65536 ≈ 1.526 × 10⁻⁵",
  ], 5.25, 1.85, 4.1, 1.15, { fontSize: 11 });

  // FSM Table
  card(s, 0.5, 3.2, 9.0, 1.95, { fill: C.card });
  s.addText("FSM Complexity Comparison", {
    x: 0.65, y: 3.28, w: 8.0, h: 0.28, fontSize: 10, color: C.primary, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  const tableData = [
    [
      { text: "Algorithm", options: { bold: true, fill: { color: C.surface }, color: C.text, fontSize: 11, fontFace: "Helvetica" } },
      { text: "FSM States", options: { bold: true, fill: { color: C.surface }, color: C.text, fontSize: 11, fontFace: "Helvetica" } },
      { text: "Convergence", options: { bold: true, fill: { color: C.surface }, color: C.text, fontSize: 11, fontFace: "Helvetica" } },
      { text: "Core Hardware", options: { bold: true, fill: { color: C.surface }, color: C.text, fontSize: 11, fontFace: "Helvetica" } },
    ],
    [
      { text: "Restoring", options: { color: C.primary, bold: true } },
      "7", "Linear  O(n)", "Subtractor + Shift Registers",
    ],
    [
      { text: "Newton-Raphson", options: { color: C.teal, bold: true } },
      "6", "Quadratic  O(log n)", "Multiplier + 256-word LUT",
    ],
    [
      { text: "Goldschmidt", options: { color: C.violet, bold: true } },
      "5", "Quadratic  O(log n)", "Parallel Multipliers + LUT",
    ],
  ];
  s.addTable(tableData, {
    x: 0.65, y: 3.6, w: 8.7, colW: [1.8, 1.4, 2.1, 3.4],
    border: { pt: 0.4, color: C.border },
    fontSize: 11, fontFace: "Helvetica", color: C.text,
    rowH: 0.36,
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 9 — ASIC SYNTHESIS RESULTS
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Chapter 4", "ASIC Synthesis Results — Cadence Genus");

  // Chart
  card(s, 0.5, 1.2, 5.3, 3.95);
  s.addText("Normalized Cell Area & Power", {
    x: 0.65, y: 1.28, w: 5.0, h: 0.28, fontSize: 10, color: C.primary, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  s.addChart(pres.charts.BAR, [
    { name: "Cell Area (×100 µm²)", labels: ["Restoring", "Newton-Raphson", "Goldschmidt"], values: [49.6, 124.5, 158.9] },
    { name: "Power (×0.1 mW)", labels: ["Restoring", "Newton-Raphson", "Goldschmidt"], values: [12.65, 48.2, 61.4] },
  ], {
    x: 0.6, y: 1.6, w: 5.1, h: 3.45,
    barDir: "col", barGrouping: "clustered",
    chartColors: [C.primary, C.teal],
    chartArea: { fill: { color: C.bg } },
    plotArea: { fill: { color: C.bg } },
    catAxisLabelColor: C.subtext, catAxisLabelFontSize: 10, catAxisLabelFontFace: "Helvetica",
    valAxisLabelColor: C.subtext, valAxisLabelFontSize: 9,
    valGridLine: { color: C.hairline, size: 0.5 },
    catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendColor: C.subtext, legendFontSize: 10, legendFontFace: "Helvetica",
    showValue: true, dataLabelColor: C.text, dataLabelFontSize: 8,
    barGapWidthPct: 60,
  });

  // Detailed metrics table
  card(s, 6.0, 1.2, 3.5, 3.95);
  s.addText("Detailed Metrics", {
    x: 6.15, y: 1.28, w: 3.2, h: 0.28, fontSize: 10, color: C.primary, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  const headers = [
    { text: "Metric", options: { bold: true, fill: { color: C.surface }, color: C.text, fontSize: 9, fontFace: "Helvetica" } },
    { text: "REST", options: { bold: true, fill: { color: C.primaryL }, color: C.primary, fontSize: 9, align: "center" } },
    { text: "NR", options: { bold: true, fill: { color: C.tealL }, color: C.teal, fontSize: 9, align: "center" } },
    { text: "GOLD", options: { bold: true, fill: { color: C.violetL }, color: C.violet, fontSize: 9, align: "center" } },
  ];
  const rows = [
    ["Cell Count", "543", "1,842", "2,156"],
    ["Cell Area (µm²)", "4,963", "12,450", "15,890"],
    ["Crit. Path (ns)", "3.888", "3.001", "2.400"],
    ["Max Freq (MHz)", "250.0", "333.3", "416.6"],
    ["Power (mW)", "1.265", "4.820", "6.140"],
    ["Latency (cyc)", "32", "6", "4"],
    ["TPut (MSPS)", "7.81", "55.55", "104.15"],
  ];
  const tdata = [headers, ...rows.map(r => [
    { text: r[0], options: { fontSize: 9, color: C.text, fontFace: "Helvetica" } },
    { text: r[1], options: { fontSize: 9, align: "center", color: C.text, fontFace: "Helvetica" } },
    { text: r[2], options: { fontSize: 9, align: "center", color: C.text, fontFace: "Helvetica" } },
    { text: r[3], options: { fontSize: 9, align: "center", color: C.text, fontFace: "Helvetica" } },
  ])];
  s.addTable(tdata, {
    x: 6.1, y: 1.6, w: 3.3, colW: [1.35, 0.65, 0.65, 0.65],
    border: { pt: 0.3, color: C.border },
    rowH: 0.32,
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 10 — RTL SIMULATION
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Chapter 6.2", "RTL Simulation & Functional Verification");

  // Waveform image
  card(s, 0.5, 1.2, 9.0, 2.55);
  s.addText("RTL Waveform — Q16.16 Fixed-Point Behaviour (Vivado)", {
    x: 0.65, y: 1.28, w: 8.7, h: 0.28, fontSize: 10, color: C.primary, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  s.addImage({ path: `${ASSET}/rtl_waveform.jpg`, x: 0.6, y: 1.6, w: 8.8, h: 2.1, sizing: { type: "contain", w: 8.8, h: 2.1 } });

  // Three result cards
  const verify = [
    { title: "Restoring", color: C.primary, lcolor: C.primaryL, text: "Sequential 32-cycle execution. Waveform confirms shift-subtract semantics.", tag: "32 cycles" },
    { title: "Newton-Raphson", color: C.teal, lcolor: C.tealL, text: "Fast convergence within 8–12 cycles using x₊₁ = x(2 − Dx).", tag: "8–12 cycles" },
    { title: "Goldschmidt", color: C.violet, lcolor: C.violetL, text: "Parallel N & D updates enable high-speed convergence — all signed cases pass.", tag: "8–12 cycles" },
  ];
  verify.forEach((v, i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 3.85, 2.95, 1.3);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 3.85, w: 0.05, h: 1.3, fill: { color: v.color }, line: { color: v.color } });
    s.addText(v.title, { x: x + 0.15, y: 3.92, w: 1.8, h: 0.3, fontSize: 12, bold: true, color: v.color, fontFace: "Helvetica" });
    pill(s, x + 1.95, 3.95, 0.95, 0.25, v.tag, v.color);
    s.addText(v.text, { x: x + 0.15, y: 4.25, w: 2.7, h: 0.85, fontSize: 10, color: C.subtext, valign: "top", fontFace: "Helvetica" });
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 11 — STANDALONE IP CORE — SCHEMATICS & RESOURCES
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Chapter 6", "FPGA Standalone IP Core — Block Schematics");

  const items = [
    { name: "Restoring", color: C.primary, img: "rest_standalone.jpg" },
    { name: "Newton-Raphson", color: C.teal, img: "nr_standalone.jpg" },
    { name: "Goldschmidt", color: C.violet, img: "gold_standalone.jpg" },
  ];
  items.forEach((it, i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 1.2, 2.95, 3.95);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.2, w: 2.95, h: 0.4, fill: { color: it.color }, line: { color: it.color } });
    s.addText(it.name, { x: x + 0.15, y: 1.2, w: 2.7, h: 0.4, fontSize: 13, bold: true, color: C.bg, valign: "middle", fontFace: "Helvetica" });
    s.addImage({ path: `${ASSET}/${it.img}`, x: x + 0.05, y: 1.7, w: 2.85, h: 1.6, sizing: { type: "contain", w: 2.85, h: 1.6 } });

    // Resource numbers
    const data = i === 0 ? ["412", "135", "0", "228.3 MHz", "0.082 W"]
              : i === 1 ? ["890", "450", "12", "318.4 MHz", "0.145 W"]
              :           ["1120", "580", "16", "350.8 MHz", "0.178 W"];
    const labels = ["LUTs", "FFs", "DSP", "Fmax", "Power"];
    labels.forEach((lab, j) => {
      const ly = 3.4 + j * 0.32;
      s.addText(lab, { x: x + 0.15, y: ly, w: 1.0, h: 0.28, fontSize: 10, color: C.muted, fontFace: "Helvetica", valign: "middle" });
      s.addText(data[j], { x: x + 1.1, y: ly, w: 1.75, h: 0.28, fontSize: 11, bold: true, color: it.color, fontFace: "Helvetica", align: "right", valign: "middle" });
    });
  });

  s.addText("Target Device: Xilinx Artix-7 (xc7a100tcsg324-1)  ·  Tool: Vivado 2023.1  ·  Format: Q16.16", {
    x: 0.5, y: 5.1, w: 9.0, h: 0.22, fontSize: 9, italic: true, color: C.muted, align: "center", fontFace: "Helvetica",
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 12 — FPGA RESOURCE / FMAX CHART
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Chapter 6.3", "FPGA Standalone IP — Resource & Timing");

  card(s, 0.5, 1.2, 5.8, 3.95);
  s.addText("Resource Utilisation (Artix-7)", {
    x: 0.65, y: 1.28, w: 5.5, h: 0.28, fontSize: 10, color: C.primary, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  s.addChart(pres.charts.BAR, [
    { name: "LUTs", labels: ["Restoring", "Newton-Raphson", "Goldschmidt"], values: [412, 890, 1120] },
    { name: "Flip-Flops", labels: ["Restoring", "Newton-Raphson", "Goldschmidt"], values: [135, 450, 580] },
    { name: "DSP Blocks", labels: ["Restoring", "Newton-Raphson", "Goldschmidt"], values: [0, 12, 16] },
  ], {
    x: 0.6, y: 1.65, w: 5.6, h: 3.4,
    barDir: "col", barGrouping: "clustered",
    chartColors: [C.primary, C.teal, C.amber],
    chartArea: { fill: { color: C.bg } },
    plotArea: { fill: { color: C.bg } },
    catAxisLabelColor: C.subtext, catAxisLabelFontSize: 10, catAxisLabelFontFace: "Helvetica",
    valAxisLabelColor: C.subtext, valAxisLabelFontSize: 9,
    valGridLine: { color: C.hairline, size: 0.5 },
    catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendColor: C.subtext, legendFontSize: 10,
    showValue: true, dataLabelColor: C.text, dataLabelFontSize: 8,
  });

  // Fmax callouts
  const callouts = [
    { label: "Restoring", val: "228.3 MHz", sub: "0.082 W · 32 cycles", color: C.primary, lc: C.primaryL },
    { label: "Newton-Raphson", val: "318.4 MHz", sub: "0.145 W · 8–12 cycles", color: C.teal, lc: C.tealL },
    { label: "Goldschmidt", val: "350.8 MHz", sub: "0.178 W · 8–12 cycles", color: C.violet, lc: C.violetL },
  ];
  callouts.forEach((c, i) => {
    const y = 1.2 + i * 1.32;
    card(s, 6.5, y, 3.0, 1.2);
    s.addShape(pres.shapes.RECTANGLE, { x: 6.5, y, w: 0.06, h: 1.2, fill: { color: c.color }, line: { color: c.color } });
    s.addText(c.label, { x: 6.6, y: y + 0.08, w: 2.85, h: 0.24, fontSize: 10, color: C.subtext, fontFace: "Helvetica" });
    s.addText(c.val, { x: 6.6, y: y + 0.32, w: 2.85, h: 0.45, fontSize: 18, bold: true, color: c.color, fontFace: "Helvetica" });
    s.addText(c.sub, { x: 6.6, y: y + 0.78, w: 2.85, h: 0.28, fontSize: 10, color: C.muted, fontFace: "Helvetica" });
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 13 — VIO INTEGRATION FLOW + SCHEMATICS
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Chapter 7", "VIO-Integrated Hardware Verification");

  // VIO flow strip
  card(s, 0.5, 1.2, 9.0, 0.85, { fill: C.surface });
  s.addText("VIO Integration Flow", { x: 0.65, y: 1.28, w: 3.0, h: 0.28, fontSize: 10, color: C.primary, bold: true, charSpacing: 2, fontFace: "Helvetica" });
  const steps = ["Reset", "Inject D & V", "Trigger start", "Wait done", "Read quotient"];
  steps.forEach((st, i) => {
    const x = 0.65 + i * 1.78;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w: 1.55, h: 0.38, rectRadius: 0.05, fill: { color: C.bg }, line: { color: C.border, width: 0.5 } });
    s.addText(st, { x, y: 1.6, w: 1.55, h: 0.38, fontSize: 10, color: C.text, align: "center", valign: "middle", fontFace: "Helvetica" });
    if (i < steps.length - 1) {
      s.addText("→", { x: x + 1.55, y: 1.6, w: 0.23, h: 0.38, fontSize: 14, color: C.muted, align: "center", valign: "middle", fontFace: "Helvetica" });
    }
  });

  // Three schematics with metrics
  const items = [
    { name: "Restoring", color: C.primary, img: "rest_vio.jpg", luts: "692", regs: "450", dsp: "0", fmax: "228.3 MHz", power: "0.086 W", lat: "32 cyc" },
    { name: "Newton-Raphson", color: C.teal, img: "nr_vio.jpg", luts: "1170", regs: "765", dsp: "12", fmax: "318.4 MHz", power: "0.150 W", lat: "8–12 cyc" },
    { name: "Goldschmidt", color: C.violet, img: "gold_vio.jpg", luts: "1400", regs: "895", dsp: "16", fmax: "350.8 MHz", power: "0.183 W", lat: "8–12 cyc" },
  ];
  items.forEach((it, i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 2.2, 2.95, 2.95);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 2.2, w: 2.95, h: 0.36, fill: { color: it.color }, line: { color: it.color } });
    s.addText(it.name, { x: x + 0.15, y: 2.2, w: 2.7, h: 0.36, fontSize: 12, bold: true, color: C.bg, valign: "middle", fontFace: "Helvetica" });

    s.addImage({ path: `${ASSET}/${it.img}`, x: x + 0.05, y: 2.6, w: 2.85, h: 1.2, sizing: { type: "contain", w: 2.85, h: 1.2 } });

    const data = [["LUTs", it.luts], ["Regs", it.regs], ["DSP", it.dsp], ["Fmax", it.fmax], ["Power", it.power], ["Latency", it.lat]];
    data.forEach(([k, v], j) => {
      const col = j % 2;
      const row = Math.floor(j / 2);
      const dx = x + 0.15 + col * 1.4;
      const dy = 3.85 + row * 0.4;
      s.addText(k, { x: dx, y: dy, w: 0.55, h: 0.32, fontSize: 9, color: C.muted, fontFace: "Helvetica", valign: "middle" });
      s.addText(v, { x: dx + 0.55, y: dy, w: 0.85, h: 0.32, fontSize: 10, bold: true, color: it.color, fontFace: "Helvetica", valign: "middle" });
    });
  });

  s.addText("All three designs verified on actual silicon — 100% accuracy under real timing/routing/clock conditions", {
    x: 0.5, y: 5.2, w: 9.0, h: 0.22, fontSize: 9, italic: true, color: C.muted, align: "center", fontFace: "Helvetica",
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 14 — PHYSICAL HARDWARE DEMO
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Chapter 8", "Physical Hardware Demonstration — ZedBoard (XC7Z020)");

  // ZedBoard photo card
  card(s, 0.5, 1.2, 4.0, 3.95);
  s.addText("Zynq-7000 SoC Development Board", {
    x: 0.65, y: 1.28, w: 3.7, h: 0.28, fontSize: 10, color: C.primary, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  s.addImage({ path: `${ASSET}/zedboard.jpg`, x: 0.7, y: 1.6, w: 3.6, h: 2.4, sizing: { type: "contain", w: 3.6, h: 2.4 } });
  bullets(s, [
    "Dual-core ARM Cortex-A9 (PS) + PL fabric",
    "DSP48E1 blocks for multiplier acceleration",
    "100 MHz system clock · JTAG debug",
    "VIO + ILA cores for in-circuit verification",
  ], 0.7, 4.05, 3.6, 1.05, { fontSize: 10 });

  // Hardware demo photos
  card(s, 4.7, 1.2, 4.8, 3.95);
  s.addText("Live VIO Hardware Verification", {
    x: 4.85, y: 1.28, w: 4.5, h: 0.28, fontSize: 10, color: C.violet, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  s.addImage({ path: `${ASSET}/hw_demo_nr.jpg`, x: 4.85, y: 1.6, w: 4.5, h: 2.0, sizing: { type: "contain", w: 4.5, h: 2.0 } });

  // Result summary mini-table
  const hwRows = [
    ["Restoring", "32 cyc", "PASS", C.primary],
    ["Newton-Raphson", "8–12 cyc", "PASS", C.teal],
    ["Goldschmidt", "8–12 cyc", "PASS", C.violet],
  ];
  hwRows.forEach(([algo, lat, status, color], i) => {
    const y = 3.7 + i * 0.45;
    s.addShape(pres.shapes.RECTANGLE, { x: 4.85, y, w: 0.05, h: 0.4, fill: { color }, line: { color } });
    s.addText(algo, { x: 4.95, y, w: 1.7, h: 0.4, fontSize: 11, bold: true, color, fontFace: "Helvetica", valign: "middle" });
    s.addText(lat, { x: 6.65, y, w: 1.5, h: 0.4, fontSize: 10, color: C.text, fontFace: "Helvetica", valign: "middle" });
    pill(s, 8.2, y + 0.06, 1.0, 0.28, status, C.green);
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 15 — ADVANTAGES & LIMITATIONS
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Trade-offs", "Advantages & Limitations per Architecture");

  const algos = [
    {
      name: "Restoring", color: C.primary, lcolor: C.primaryL,
      adv: ["Minimal silicon area — no DSP blocks", "Lowest power (1.265 mW ASIC)", "Simple FSM, easy to verify", "Predictable 32-cycle latency"],
      lim: ["Highest latency (32 cycles)", "Throughput only 7.81 MSPS", "Sequential — cannot pipeline"],
    },
    {
      name: "Newton-Raphson", color: C.teal, lcolor: C.tealL,
      adv: ["Quadratic convergence — 6× faster", "Balanced area/power/speed", "55.55 MSPS throughput", "Industry-standard CPU choice"],
      lim: ["Requires 32×32 multiplier", "LUT for seed adds area", "Approximation needs careful bit-width"],
    },
    {
      name: "Goldschmidt", color: C.violet, lcolor: C.violetL,
      adv: ["Highest Fmax — 416.6 MHz (ASIC)", "Best throughput — 104.15 MSPS", "Parallel paths shrink cycles", "Ideal for DSP / AI accelerators"],
      lim: ["Highest power (6.14 mW ASIC)", "Largest area (15,890 µm²)", "Greater verification complexity"],
    },
  ];

  algos.forEach((a, i) => {
    const x = 0.5 + i * 3.05;
    // Header
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.2, w: 2.95, h: 0.4, fill: { color: a.color }, line: { color: a.color } });
    s.addText(a.name, { x, y: 1.2, w: 2.95, h: 0.4, fontSize: 13, bold: true, color: C.bg, align: "center", valign: "middle", fontFace: "Helvetica" });

    // Advantages section
    card(s, x, 1.6, 2.95, 1.85, { fill: C.greenL, border: C.green });
    s.addText("✓  ADVANTAGES", { x: x + 0.12, y: 1.65, w: 2.7, h: 0.22, fontSize: 9, bold: true, color: C.green, charSpacing: 2, fontFace: "Helvetica" });
    bullets(s, a.adv, x + 0.12, 1.88, 2.7, 1.55, { fontSize: 10 });

    // Limitations
    card(s, x, 3.5, 2.95, 1.65, { fill: C.roseL, border: C.rose });
    s.addText("✗  LIMITATIONS", { x: x + 0.12, y: 3.55, w: 2.7, h: 0.22, fontSize: 9, bold: true, color: C.rose, charSpacing: 2, fontFace: "Helvetica" });
    bullets(s, a.lim, x + 0.12, 3.78, 2.7, 1.35, { fontSize: 10 });
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 16 — COMPARATIVE ANALYSIS
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Summary", "Comparative Analysis & Best-Fit Recommendation");

  // Throughput chart
  card(s, 0.5, 1.2, 4.4, 2.85);
  s.addText("Throughput  (MSPS, ASIC)", {
    x: 0.65, y: 1.28, w: 4.0, h: 0.28, fontSize: 10, color: C.primary, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  s.addChart(pres.charts.BAR, [{
    name: "Throughput (MSPS)",
    labels: ["Restoring", "Newton-Raphson", "Goldschmidt"],
    values: [7.81, 55.55, 104.15]
  }], {
    x: 0.6, y: 1.6, w: 4.2, h: 2.4,
    barDir: "col",
    chartColors: [C.primary, C.teal, C.violet],
    chartColorsOpacity: 80,
    chartArea: { fill: { color: C.bg } }, plotArea: { fill: { color: C.bg } },
    catAxisLabelColor: C.subtext, catAxisLabelFontSize: 10,
    valAxisLabelColor: C.subtext, valAxisLabelFontSize: 9,
    valGridLine: { color: C.hairline, size: 0.5 }, catGridLine: { style: "none" },
    showValue: true, dataLabelColor: C.text, dataLabelFontSize: 9,
    showLegend: false,
    barGapWidthPct: 90,
    showTitle: false,
  });

  // Power chart
  card(s, 5.1, 1.2, 4.4, 2.85);
  s.addText("Power  (mW, ASIC)", {
    x: 5.25, y: 1.28, w: 4.0, h: 0.28, fontSize: 10, color: C.primary, bold: true, charSpacing: 2, fontFace: "Helvetica",
  });
  s.addChart(pres.charts.BAR, [{
    name: "Total Power (mW)",
    labels: ["Restoring", "Newton-Raphson", "Goldschmidt"],
    values: [1.265, 4.820, 6.140]
  }], {
    x: 5.2, y: 1.6, w: 4.2, h: 2.4,
    barDir: "col",
    chartColors: [C.primary, C.teal, C.violet],
    chartArea: { fill: { color: C.bg } }, plotArea: { fill: { color: C.bg } },
    catAxisLabelColor: C.subtext, catAxisLabelFontSize: 10,
    valAxisLabelColor: C.subtext, valAxisLabelFontSize: 9,
    valGridLine: { color: C.hairline, size: 0.5 }, catGridLine: { style: "none" },
    showValue: true, dataLabelColor: C.text, dataLabelFontSize: 9,
    showLegend: false, barGapWidthPct: 90,
  });

  // Best-fit cards
  const summary = [
    { label: "Low-Power IoT", winner: "Restoring", color: C.primary, lc: C.primaryL },
    { label: "Balanced Performance", winner: "Newton-Raphson", color: C.teal, lc: C.tealL },
    { label: "High-Speed DSP / AI", winner: "Goldschmidt", color: C.violet, lc: C.violetL },
  ];
  summary.forEach((it, i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 4.15, 2.95, 1.0, { fill: it.lc, border: it.color });
    s.addText("BEST FOR", { x: x + 0.15, y: 4.22, w: 2.7, h: 0.22, fontSize: 9, color: it.color, bold: true, charSpacing: 3, fontFace: "Helvetica" });
    s.addText(it.label, { x: x + 0.15, y: 4.42, w: 2.7, h: 0.3, fontSize: 12, color: C.text, fontFace: "Helvetica" });
    s.addText(it.winner, { x: x + 0.15, y: 4.72, w: 2.7, h: 0.38, fontSize: 16, bold: true, color: it.color, fontFace: "Helvetica" });
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 17 — KEY CONTRIBUTIONS
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Contributions", "Key Contributions of this Work");

  const contributions = [
    { num: "01", title: "Unified Benchmarking Framework", desc: "First direct apples-to-apples ASIC comparison of three 32-bit fixed-point dividers under identical synthesis constraints — fills a gap in 64-bit FPU literature.", color: C.primary, lc: C.primaryL },
    { num: "02", title: "Signed Fixed-Point Wrapper", desc: "Reusable Two's Complement pre/post-processing wrapper using XOR sign logic — enables all three unsigned cores to handle signed Q16.16 arithmetic.", color: C.teal, lc: C.tealL },
    { num: "03", title: "Silicon-Validated Hardware", desc: "All three designs deployed on Zynq-7000 SoC and verified at 100% numerical accuracy via VIO real-time hardware-in-the-loop testing.", color: C.violet, lc: C.violetL },
    { num: "04", title: "Full VLSI Design Flow", desc: "Complete RTL → Cadence Genus ASIC synthesis + Vivado FPGA implementation → VIO verification pipeline executed for each architecture.", color: C.amber, lc: C.amberL },
  ];

  contributions.forEach((c, i) => {
    const row = Math.floor(i / 2);
    const col = i % 2;
    const x = 0.5 + col * 4.55;
    const y = 1.2 + row * 1.95;
    card(s, x, y, 4.45, 1.8);
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.06, h: 1.8, fill: { color: c.color }, line: { color: c.color } });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.2, y: y + 0.18, w: 0.55, h: 0.55, rectRadius: 0.08, fill: { color: c.lc }, line: { color: c.lc } });
    s.addText(c.num, { x: x + 0.2, y: y + 0.18, w: 0.55, h: 0.55, fontSize: 13, bold: true, color: c.color, align: "center", valign: "middle", fontFace: "Helvetica" });
    s.addText(c.title, { x: x + 0.85, y: y + 0.2, w: 3.5, h: 0.45, fontSize: 13, bold: true, color: C.text, fontFace: "Helvetica" });
    s.addText(c.desc, { x: x + 0.2, y: y + 0.82, w: 4.15, h: 0.95, fontSize: 10.5, color: C.subtext, valign: "top", fontFace: "Helvetica" });
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 18 — CONCLUSION & FUTURE WORK
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Closing", "Conclusion & Future Scope");

  // Conclusion
  card(s, 0.5, 1.2, 5.4, 3.95, { fill: C.surface });
  s.addText("Conclusion", { x: 0.7, y: 1.28, w: 5.0, h: 0.32, fontSize: 14, bold: true, color: C.primary, fontFace: "Helvetica" });
  bullets(s, [
    "Restoring  ·  Optimal for area-constrained systems. Zero DSP, 32-cycle latency.",
    "Newton-Raphson  ·  Best design balance. Quadratic convergence, moderate overhead.",
    "Goldschmidt  ·  Premier choice for HPC / DSP / AI. 416.6 MHz, 104 MSPS.",
    "All designs achieve 100% accuracy in Q16.16 fixed-point on real silicon.",
    "VIO verification confirms silicon-readiness for production deployment.",
  ], 0.7, 1.7, 5.0, 3.4, { fontSize: 11.5 });

  // Future Scope
  card(s, 6.1, 1.2, 3.4, 3.95);
  s.addText("Future Scope", { x: 6.3, y: 1.28, w: 3.0, h: 0.32, fontSize: 14, bold: true, color: C.amber, fontFace: "Helvetica" });

  const future = [
    ["Variable Latency", "Optimize average-case via early termination", C.primary],
    ["Approximate Division", "Error-tolerant designs for deep learning", C.teal],
    ["IEEE 754 FPU", "Migrate core logic to floating-point unit", C.violet],
    ["High-Radix (Radix-4/8)", "Further reduce iteration count", C.amber],
  ];
  future.forEach(([title, desc, color], i) => {
    const y = 1.7 + i * 0.85;
    s.addShape(pres.shapes.RECTANGLE, { x: 6.3, y, w: 0.05, h: 0.7, fill: { color }, line: { color } });
    s.addText(title, { x: 6.42, y, w: 2.95, h: 0.28, fontSize: 11, bold: true, color: C.text, fontFace: "Helvetica" });
    s.addText(desc, { x: 6.42, y: y + 0.3, w: 2.95, h: 0.4, fontSize: 9.5, color: C.subtext, fontFace: "Helvetica" });
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 19 — REFERENCES
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  sectionTitle(s, "Bibliography", "References");

  const refs = [
    "[1] R. K. L. Trummer, 'A high-performance data-dependent hardware integer divider,' M.S. thesis, Paris Lodron Univ., Salzburg, May 2005.",
    "[2] D. G. Bailey, 'Space efficient division on FPGAs,' Proc. Electronics NZ Conf., 2006, pp. 206–211.",
    "[3] J. Kumari and M. Y. Yasin, 'Design and Soft Implementation of N-bit SRT Divider on FPGA,' Int. J. Innov. Eng., vol. 3, no. 4, 2015.",
    "[4] K. Narendra et al., 'FPGA implementation of fixed point integer divider using iterative array structure,' Int. J. Eng. Tech. Res., vol. 3, no. 4, 2015.",
    "[5] E. Matthews et al., 'Rethinking integer divider design for FPGA soft-processors,' IEEE FCCM, Apr. 2019, doi: 10.1109/FCCM.2019.00046.",
    "[6] U. S. Patankar and A. Koel, 'Review of Basic Classes of Dividers Based on Division Algorithm,' IEEE Access, vol. 9, 2021.",
    "[7] K. Tatas et al., 'A novel division algorithm for parallel and sequential processing,' Proc. 9th Int. Conf. ECAS, 2002, pp. 553–556.",
    "[8] Merriam-Webster Dictionary & Cambridge Dictionary (general arithmetic definitions), accessed 2020.",
  ];

  refs.forEach((ref, i) => {
    const y = 1.15 + i * 0.5;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: y + 0.1, w: 0.04, h: 0.28, fill: { color: C.primary }, line: { color: C.primary } });
    s.addText(ref, { x: 0.65, y, w: 8.85, h: 0.5, fontSize: 10.5, color: C.text, fontFace: "Helvetica", valign: "middle" });
  });
}

// ════════════════════════════════════════════════════════════════
// SLIDE 20 — THANK YOU
// ════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  // Decorative shapes
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.05, fill: { color: C.primary }, line: { color: C.primary } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.575, w: 10, h: 0.05, fill: { color: C.primary }, line: { color: C.primary } });

  // Soft accent dots
  s.addShape(pres.shapes.OVAL, { x: 8.5, y: 0.5, w: 1.4, h: 1.4, fill: { color: C.primaryL }, line: { color: C.primaryL } });
  s.addShape(pres.shapes.OVAL, { x: 0.2, y: 4.0, w: 1.0, h: 1.0, fill: { color: C.tealL }, line: { color: C.tealL } });
  s.addShape(pres.shapes.OVAL, { x: 8.8, y: 4.4, w: 0.7, h: 0.7, fill: { color: C.amberL }, line: { color: C.amberL } });

  s.addText("Thank You", {
    x: 0.5, y: 1.6, w: 9, h: 1.0,
    fontSize: 50, fontFace: "Helvetica", bold: true,
    color: C.text, align: "center",
  });
  s.addText("Questions & Discussion", {
    x: 0.5, y: 2.7, w: 9, h: 0.5,
    fontSize: 18, fontFace: "Helvetica", color: C.primary, align: "center",
  });

  const contacts = [
    "K Francis Angel Manohar  ·  Roll 230104004",
    "Manjunath  ·  Roll 230102073",
    "Supervisor: Dr. Dilip Singh, Asst. Professor, ECE",
    "IIIT Senapati, Manipur  ·  April 2026",
  ];
  contacts.forEach((line, i) => {
    s.addText(line, {
      x: 0.5, y: 3.6 + i * 0.32, w: 9, h: 0.3,
      fontSize: 12, color: C.subtext, fontFace: "Helvetica", align: "center",
    });
  });
}

// ─── WRITE ────────────────────────────────────────────────────────
const outputPath = "/Users/suhasdev/Downloads/32bit_divider_presentation.pptx";
pres.writeFile({ fileName: outputPath })
  .then(name => console.log("PPT generated: " + name))
  .catch(err => console.error("Error:", err));
