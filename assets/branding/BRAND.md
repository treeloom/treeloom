# Treeloom Brand Guidelines

## Logo

The Treeloom mark is the **Thread-Tree**: a woven canopy symbol combining vertical
warp strands (trunk) with horizontal weft arcs (canopy), representing the two-stage
pipeline: *parse the trees, weave the graph*.

### Primary Lockup

Horizontal: mark left, wordmark right. Use in headers, nav bars, wide spaces.

- Dark: `assets/branding/treeloom-logo-horizontal.svg` (PNG: `treeloom-logo-horizontal.png`, 1600x320)
- Light: `assets/branding/treeloom-logo-horizontal-light.svg` (PNG: `treeloom-logo-horizontal-light.png`, 1600x320)

### Stacked Lockup

Mark above wordmark. Use in square spaces: app icons, avatars, profile photos.

- `assets/branding/treeloom-logo-stacked.svg` (PNG: `treeloom-logo-stacked.png`, 800x880)

### Mark Only

Standalone symbol. Use for favicons, app icons, social avatars.

- `assets/branding/treeloom-mark.svg` (PNG: `treeloom-mark.png`, 208x256)
- Favicon: `assets/branding/treeloom-favicon.svg` (32x32, dark bg, rounded; PNG: `treeloom-favicon.png`, 128x128)

### PNG Versions

Every SVG has a PNG beside it with the same name, for places that can't take
SVG (slides, social cards, chat avatars, some docs tools). They are rendered at
4x the SVG's native size on a transparent background. The dark-variant logos
have white text, so place them on a dark background, the same as the SVGs.
The SVGs remain the source of truth: prefer them wherever SVG works, and
re-export the PNGs whenever an SVG changes.

When re-exporting, render with the real **Inter** font (weights 400 and 600). A
converter that silently substitutes a system font (e.g. Noto Sans) produces a
wrong wordmark with no error. Any SVG renderer works if it is given Inter's
font files and system-font fallback is disabled; `@resvg/resvg-js` with
`font.fontFiles` set to Inter-Regular/SemiBold and `loadSystemFonts: false`
at `fitTo: { mode: 'zoom', value: 4 }` reproduces the committed PNGs.

## Color Palette

| Role | Hex | Usage |
|------|-----|-------|
| Primary Green | `#3fb950` | Main structural elements, primary buttons |
| Bright Green | `#39d353` | Highlights, active states, canopy arcs |
| Deep Green | `#2ea043` | Deep structure, bottom canopy layer |
| Amber | `#d29922` | Connection nodes, weave intersections, accents |
| Dark BG | `#0d1117` | Primary background (dark mode) |
| Surface | `#161b22` | Card backgrounds |
| Text | `#c9d1d9` | Primary text on dark |
| Muted | `#8b949e` | Secondary text, taglines |

## Typography

- **Wordmark**: Inter, weight 600, letter-spacing -0.03em
- **Tagline**: Inter, weight 400, letter-spacing +0.06em, muted color
- **UI text**: Inter, system-ui, -apple-system, sans-serif stack

## Tagline

**Parse Trees · Weave Graphs**

Always presented in all-caps, muted, below the wordmark.

## Usage Notes

- Maintain clear space around the mark equal to the height of the 't' in treeloom
- Never recolor the mark (green + amber are fixed)
- On light backgrounds, use the light variant (greens shift to higher contrast)
- Minimum logo width: 120px for horizontal lockup
- Favicon: use the 32x32 SVG or generate a multi-size .ico
