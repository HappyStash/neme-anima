/** Per-character color palette so badges/chips for different characters in
 *  the same project are immediately tellable apart. Colors are assigned by
 *  the character's index in the project's `characters` list, cycling through
 *  the palette — order matches the order returned by the API, which is
 *  stable across reloads, so a given character keeps the same color for the
 *  life of the project (until someone deletes a character before it).
 *
 *  Class strings are written out as literals so Tailwind's content scanner
 *  picks them up at build time. Don't construct them with template strings
 *  (e.g. ``bg-${name}-500``) — those won't survive the purge. */

export type CharacterColor = {
  /** Solid 500 fill — used for the active chip background. */
  bgActive: string;
  /** 400 border that pairs with the 500 fill. */
  borderActive: string;
  /** 80% alpha 500 fill — used for the smaller FrameThumb badge so the
   *  underlying image still bleeds through a touch. */
  bgSoft: string;
  /** Smaller 400 swatch for the dot we paint next to inactive chips so
   *  the user can still tell who's who without selecting the chip. */
  dot: string;
  /** Glow rgba for the box-shadow on the active chip. Inline-styled. */
  glow: string;
  /** Solid rgba for full-opacity outlines (e.g. the active-ref ring on
   *  the per-video strip). Same hue as `glow` but at full alpha. */
  ring: string;
};

const PALETTE: CharacterColor[] = [
  { bgActive: "bg-indigo-500",  borderActive: "border-indigo-400",  bgSoft: "bg-indigo-500/80",  dot: "bg-indigo-400",  glow: "rgba(99,102,241,0.45)",  ring: "rgba(99,102,241,1)" },
  { bgActive: "bg-emerald-500", borderActive: "border-emerald-400", bgSoft: "bg-emerald-500/80", dot: "bg-emerald-400", glow: "rgba(16,185,129,0.45)", ring: "rgba(16,185,129,1)" },
  { bgActive: "bg-amber-500",   borderActive: "border-amber-400",   bgSoft: "bg-amber-500/80",   dot: "bg-amber-400",   glow: "rgba(245,158,11,0.45)", ring: "rgba(245,158,11,1)" },
  { bgActive: "bg-rose-500",    borderActive: "border-rose-400",    bgSoft: "bg-rose-500/80",    dot: "bg-rose-400",    glow: "rgba(244,63,94,0.45)",  ring: "rgba(244,63,94,1)" },
  { bgActive: "bg-sky-500",     borderActive: "border-sky-400",     bgSoft: "bg-sky-500/80",     dot: "bg-sky-400",     glow: "rgba(14,165,233,0.45)", ring: "rgba(14,165,233,1)" },
  { bgActive: "bg-fuchsia-500", borderActive: "border-fuchsia-400", bgSoft: "bg-fuchsia-500/80", dot: "bg-fuchsia-400", glow: "rgba(217,70,239,0.45)", ring: "rgba(217,70,239,1)" },
  { bgActive: "bg-lime-500",    borderActive: "border-lime-400",    bgSoft: "bg-lime-500/80",    dot: "bg-lime-400",    glow: "rgba(132,204,22,0.45)", ring: "rgba(132,204,22,1)" },
  { bgActive: "bg-orange-500",  borderActive: "border-orange-400",  bgSoft: "bg-orange-500/80",  dot: "bg-orange-400",  glow: "rgba(249,115,22,0.45)", ring: "rgba(249,115,22,1)" },
  { bgActive: "bg-cyan-500",    borderActive: "border-cyan-400",    bgSoft: "bg-cyan-500/80",    dot: "bg-cyan-400",    glow: "rgba(6,182,212,0.45)",  ring: "rgba(6,182,212,1)" },
  { bgActive: "bg-teal-500",    borderActive: "border-teal-400",    bgSoft: "bg-teal-500/80",    dot: "bg-teal-400",    glow: "rgba(20,184,166,0.45)", ring: "rgba(20,184,166,1)" },
];

/** Color for a given index. Cycles through the palette so projects with
 *  more than 10 characters reuse colors — collisions then are unavoidable
 *  but rare in practice. */
export function colorForIndex(index: number): CharacterColor {
  return PALETTE[((index % PALETTE.length) + PALETTE.length) % PALETTE.length];
}

/** Color for a slug, resolved against the project's characters list. Returns
 *  the first palette entry for orphan slugs that no longer match any current
 *  character — those frames render under "Unsorted" anyway. */
export function colorForSlug(
  slug: string,
  characters: readonly { slug: string }[],
): CharacterColor {
  const idx = characters.findIndex((c) => c.slug === slug);
  return colorForIndex(idx < 0 ? 0 : idx);
}
