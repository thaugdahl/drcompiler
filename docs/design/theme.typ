//===-- theme.typ — shared styling + cetz drawing helpers -----------------===//
// One palette + a small set of cetz node/arrow helpers so every figure in the
// drcompiler design docs reads the same.  Import with `#import "theme.typ": *`.

#import "@preview/cetz:0.3.4"

// ---- palette ------------------------------------------------------------
#let pal = (
  ink:    rgb("#1d2433"),
  muted:  rgb("#5b6577"),
  line:   rgb("#9aa4b2"),
  mm:     rgb("#2f6f8f"),   // the Machine Model (the spine)
  mmbg:   rgb("#e3eef3"),
  pass:   rgb("#6a4c93"),   // a transform pass
  passbg: rgb("#ece5f4"),
  io:     rgb("#3a7d44"),   // IR / data artifacts
  iobg:   rgb("#e4f0e6"),
  win:    rgb("#2e7d32"),   // a measured win
  loss:   rgb("#b3261e"),   // a measured no-go
  warnbg: rgb("#fdecea"),
  accent: rgb("#c65d00"),   // highlight
  accentbg: rgb("#fcecdc"),
  dec:    rgb("#8a6d00"),   // a decision / branch point
  decbg:  rgb("#fbf3d3"),
  gridln: rgb("#d6dbe3"),
)

// ---- document setup -----------------------------------------------------
#let setup(body) = {
  set page(
    paper: "a4", margin: (x: 2.2cm, y: 2.4cm),
    numbering: "1", number-align: center,
  )
  set text(font: "New Computer Modern", size: 10.5pt, fill: pal.ink)
  set par(justify: true, leading: 0.62em)
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    block(above: 0.6em, below: 0.8em, text(size: 19pt, weight: 800, fill: pal.mm, it))
  }
  show heading.where(level: 2): it => block(above: 1.1em, below: 0.6em,
    text(size: 13.5pt, weight: 700, fill: pal.ink, it))
  show heading.where(level: 3): it => block(above: 0.9em, below: 0.4em,
    text(size: 11.5pt, weight: 700, fill: pal.muted, it))
  show raw.where(block: true): it => block(
    fill: rgb("#f6f7f9"), inset: 8pt, radius: 4pt, width: 100%,
    text(size: 8.6pt, it))
  show raw.where(block: false): it => box(
    fill: rgb("#f0f1f4"), inset: (x: 3pt, y: 0pt), outset: (y: 2pt), radius: 2pt, it)
  set figure(gap: 0.9em)
  show figure.caption: it => text(size: 9pt, fill: pal.muted, it)
  body
}

// ---- callout boxes ------------------------------------------------------
#let callout(title, body, fg: pal.mm, bg: pal.mmbg) = block(
  fill: bg, inset: 9pt, radius: 4pt, width: 100%, above: 0.8em, below: 0.8em,
  stroke: (left: 2.5pt + fg),
  [#text(weight: 700, fill: fg, title)\ #body])
#let keyidea(body) = callout("Key idea", body, fg: pal.accent, bg: pal.accentbg)
#let measured(body) = callout("Measured", body, fg: pal.win, bg: pal.iobg)
#let nogo(body) = callout("No-go (measured)", body, fg: pal.loss, bg: pal.warnbg)

// Typst's rotate, captured here so it survives `import cetz.draw: *` inside a
// canvas (which shadows the global `rotate` with cetz's transform).
#let trot(angle, body) = rotate(angle, body)

// ---- cetz drawing helpers (call inside cetz.canvas) ---------------------
// A labelled rounded box node. `kind` picks the palette (mm/pass/io/plain).
#let node(pos, body, name: none, kind: "plain", w: auto, h: auto, ..rest) = {
  let (st, fl) = if kind == "mm" { (pal.mm, pal.mmbg) }
    else if kind == "pass" { (pal.pass, pal.passbg) }
    else if kind == "io" { (pal.io, pal.iobg) }
    else if kind == "accent" { (pal.accent, pal.accentbg) }
    else if kind == "decision" { (pal.dec, pal.decbg) }
    else if kind == "win" { (pal.win, pal.iobg) }
    else if kind == "loss" { (pal.loss, pal.warnbg) }
    else { (pal.muted, white) }
  cetz.draw.content(
    pos,
    box(width: w, height: h, inset: 6pt, align(center + horizon,
      text(size: 8.5pt, fill: pal.ink, body))),
    name: name, frame: "rect", fill: fl, stroke: 0.9pt + st, padding: 0.0,
    ..rest)
}

// A connector arrow between two coordinates/anchors.
#let arrow(from, to, ..rest) = cetz.draw.line(from, to,
  stroke: 0.9pt + pal.line, mark: (end: ">", fill: pal.line, scale: 0.8), ..rest)
#let darrow(from, to, ..rest) = cetz.draw.line(from, to,
  stroke: (paint: pal.line, dash: "dashed"), mark: (end: ">", fill: pal.line, scale: 0.8), ..rest)
