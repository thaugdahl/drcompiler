// lib.typ — shared template, palette, and callouts for the drcompiler docs.
// Re-exports the diagram packages so chapters only need `#import "lib.typ": *`.

#import "@preview/cetz:0.3.4"
#import "@preview/fletcher:0.5.5" as fletcher: diagram, node, edge

// ---- palette ------------------------------------------------------------
#let accent   = rgb("#1f6feb")  // blue   — structure / pipeline
#let mmteal   = rgb("#0b7261")  // teal   — the Machine Model (the spine)
#let warm     = rgb("#b54708")  // orange — decisions / caveats
#let good     = rgb("#1a7f37")  // green  — wins / confirmed
#let bad      = rgb("#cf222e")  // red    — no-go / refuted
#let ink      = rgb("#1c2024")
#let faint    = rgb("#eef2f6")
#let mmfaint  = rgb("#e6f4f0")

// ---- callouts -----------------------------------------------------------

// THE tie-back box: "everything routes through the Machine Model".
#let mm(body) = block(
  width: 100%, inset: 10pt, radius: 5pt, fill: mmfaint,
  stroke: (left: 3pt + mmteal),
  [#text(weight: "bold", fill: mmteal, size: 9pt)[⚙ MACHINE MODEL]
   #h(6pt) #text(size: 9.5pt)[#body]],
)

#let note(title, body) = block(
  width: 100%, inset: 10pt, radius: 5pt, fill: faint, stroke: (left: 3pt + accent),
  [#text(weight: "bold", fill: accent, size: 9pt)[#upper(title)] #par(body)],
)

#let verdict(ok, body) = block(
  width: 100%, inset: 9pt, radius: 5pt,
  fill: if ok { rgb("#e9f7ec") } else { rgb("#fbe9ea") },
  stroke: (left: 3pt + (if ok { good } else { bad })),
  [#text(weight: "bold", fill: if ok { good } else { bad }, size: 9pt)[#if ok {"WIN"} else {"NO-GO"}]
   #h(6pt) #text(size: 9.5pt)[#body]],
)

// figure caption helper
#let fig(content, caption) = figure(content, caption: caption)

// ---- document template --------------------------------------------------
#let drdoc(title: "", subtitle: "", author: "", body) = {
  set document(title: title, author: author)
  set page(
    paper: "a4", margin: (x: 2.4cm, y: 2.6cm),
    header: context {
      if counter(page).get().first() > 1 [
        #set text(size: 8pt, fill: rgb("#6e7781"))
        #emph(title) #h(1fr) drcompiler
        #line(length: 100%, stroke: 0.4pt + rgb("#d0d7de"))
      ]
    },
    footer: context [
      #set text(size: 8pt, fill: rgb("#6e7781"))
      #line(length: 100%, stroke: 0.4pt + rgb("#d0d7de"))
      cache-aware recomputation & codegen #h(1fr) #counter(page).display("1")
    ],
  )
  set text(font: "New Computer Modern", size: 10.5pt, fill: ink, lang: "en")
  set par(justify: true, leading: 0.62em)
  show raw.where(block: true): it => block(
    width: 100%, inset: 8pt, radius: 4pt, fill: rgb("#f6f8fa"),
    stroke: 0.5pt + rgb("#d0d7de"), text(size: 8.5pt, it),
  )
  show raw.where(block: false): it => box(
    fill: rgb("#f0f3f6"), inset: (x: 3pt, y: 0pt), outset: (y: 2pt), radius: 2pt,
    text(size: 9pt, it),
  )
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    block(above: 0pt, below: 14pt)[
      #set text(fill: accent, size: 20pt, weight: "bold")
      #if it.numbering != none [#counter(heading).display() #h(8pt)]
      #it.body
    ]
  }
  show heading.where(level: 2): it => block(above: 16pt, below: 8pt)[
    #set text(fill: ink, size: 13pt, weight: "bold"); #counter(heading).display() #h(6pt) #it.body
  ]
  show heading.where(level: 3): it => block(above: 12pt, below: 6pt)[
    #set text(fill: mmteal, size: 11pt, weight: "bold"); #it.body
  ]
  show link: it => text(fill: accent, it)

  // ---- title page ----
  set align(center + horizon)
  block[
    #text(size: 30pt, weight: "bold", fill: ink)[#title] \
    #v(6pt)
    #text(size: 14pt, fill: rgb("#57606a"))[#subtitle]
    #v(20pt)
    #cetz.canvas(length: 1cm, {
      import cetz.draw: *
      // a small "spine" glyph: the Machine Model feeding the passes
      let lvl(x, w, h, c, t) = {
        rect((x, -h/2), (x + w, h/2), fill: c, stroke: 0.6pt + white, radius: 0.08)
        content((x + w/2, 0), text(fill: white, size: 7pt, weight: "bold")[#t])
      }
      lvl(-5, 2.2, 1.0, mmteal, "Machine\nModel")
      for (i, t) in (("recompute",), ("fission",), ("register-block",), ("strength-reduce",)).enumerate() {
        let y = 1.65 - i * 1.1
        rect((-1.7, y - 0.42), (1.9, y + 0.42), fill: faint, stroke: 0.6pt + accent, radius: 0.08)
        content((0.1, y), text(fill: ink, size: 7pt)[#t.at(0)])
        line((-2.8, 0), (-1.7, y), stroke: 1pt + mmteal, mark: (end: ">"))
      }
    })
    #v(16pt)
    #text(size: 10pt, fill: rgb("#57606a"))[An out-of-tree MLIR compiler ·
      one machine description, every cost decision]
  ]
  pagebreak()

  set align(left + top)
  outline(title: [Contents], indent: auto, depth: 2)
  pagebreak()

  body
}
