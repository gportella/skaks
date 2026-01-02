Here are practical, low-risk additions that will noticeably improve strength without complicating your search. Keep everything White-centric and independent of side-to-move.

Material phase scaling: evaluation terms should be weighted differently in opening vs endgame.

Simple approach: compute a phase from piece counts (queens/rooks/minor pieces) and blend between opening and endgame weights.
Benefit: pawn structure and king safety matter more early; mobility and passed pawns more later.
Piece-square tables (PSTs): add positional bonuses per square for each piece.

Opening PSTs reward centralization (knights on c3/d4/e4/f3, bishops on long diagonals), rooks on open/semi-open files, and kings castled.
Endgame PST moves the king toward the center and pushes passed pawns.
This gives your eval “shape” and helps the engine stop suggesting aimless queen sorties.
Pawn structure: this is high impact and relatively cheap.

Doubled pawns: penalize per file per color.
Isolated pawns: penalize if a pawn has no friendly pawns on adjacent files.
Backward pawns: penalize if a pawn cannot advance because the square ahead is controlled and it’s behind neighbors.
Passed pawns: bonus scaled by rank (bigger in endgame), plus extra if supported by another pawn.
Connected pawns: small bonuses for chains and phalanxes.
Open/semi-open files: bonus rooks/queens on semi-open files (no own pawn), bigger on open files (no pawns of either side).
You already track pawn lists; build bitboards by file to compute these quickly.
Pins and skewers: you already detect pinned pieces—use them.

Penalize the value of pinned piece activity: reduce mobility of a pinned piece in the direction it’s pinned.
Increase “pressure” on pinned defenders of the king (e.g., a pinned knight in front of king is a tactical liability).
Bonus for moves that increase the number or severity of pins (this is more a search heuristic; in eval, treat pins as static liabilities/bonuses).
Hanging/undefended pieces: if a piece is attacked more times than it is defended, apply a penalty proportional to its value.

Even a coarse “undefended major/minor piece” penalty catches many blunders.
Combine with your is_square_attacked and a simple count_defenders function.
Mobility (beyond king): count legal moves for bishops, knights, rooks, queens, with modest scaling.

Scale by game phase and reduce mobility in blocked positions.
Don’t let mobility dominate material—keep weights small.
Threats and tactical motifs:

Fork potential: bonus if a move would attack two higher-value pieces next ply is tricky in static eval; instead, penalize positions where your opponent can fork you easily (two high-value pieces close and reachable by a knight/queen).
Outposts: bonuses for knights on protected outposts (enemy pawn cannot attack that square).
Bishop pair: small bonus if you have both bishops, larger in open positions/endgame.
King safety (richer):

Pawn shelter: evaluate the pawn structure in front of the king (files f–h for white short castle, a–c for long castle), penalize holes on g/h or f/g files.
Enemy attack weights around the king ring (a standard “king danger” term): count attacking pieces’ pressure on squares near the king.
Tempo and development:

Development lead: bonus for getting minor pieces out and castling earlier; small and phase-weighted.
Penalty for moving the same piece multiple times early (optional and small).
Space:

Count controlled squares in opponent half (with pawns and pieces), modest bonus. Helps discourage cramped positions.
Drawishness:

Opposite-colored bishops endgame: reduce evaluation magnitude (closer to draw).
Insufficient mating material patterns: detect and clamp scores.
Implementation tips

Keep weights conservative. It’s better to be slightly underfit than to drown material with shape terms.
Sum both sides’ contributions in White-centric style: add bonuses for White, subtract for Black.
Cache per-color pawn bitboards by file and rank to reuse across pawn structure, open files, passed pawn checks.
Make evaluation cheap: avoid calling heavy attack routines in tight loops; precompute attack masks or use incremental evaluation if you later add it.
Blend with phase: total = opening_eval * (phase) + endgame_eval * (1 - phase).
How to use your pinned-pieces function

For each pinned piece:
Reduce its mobility score significantly.
If the pin is on the king line (e.g., piece pinned to king), add extra penalty, scaled by piece value.
If the pinned piece is a defender of a critical square (e.g., mating square near king), increase penalty slightly.
For opponent pinned pieces: add the symmetric bonus for White.
