#ifndef TBCONFIG_H
#define TBCONFIG_H

/****************************************************************************/
/* BUILD CONFIG:                                                            */
/****************************************************************************/

/* Define TB_CUSTOM_POP_COUNT to override internal popcount implementation. */
/* #define TB_CUSTOM_POP_COUNT(x) <DEFINITION> */

/* Define TB_CUSTOM_LSB to override internal lsb implementation. */
/* #define TB_CUSTOM_LSB(x) <DEFINITION> */

/* Define TB_NO_STDINT if you do not want to use <stdint.h>. */
/* #define TB_NO_STDINT */

/* Define TB_NO_STDBOOL if you do not want to use <stdbool.h>. */
/* #define TB_NO_STDBOOL */

/* Define TB_NO_THREADS if your program is not multi-threaded. */
/* #define TB_NO_THREADS */

/* Disable helper API to reduce compiled surface. */
#define TB_NO_HELPER_API

/* Define TB_NO_HW_POP_COUNT if no hardware popcount. */
/* #define TB_NO_HW_POP_COUNT */

/***************************************************************************/
/* SCORING CONSTANTS                                                       */
/***************************************************************************/
#define TB_VALUE_PAWN 100
#define TB_VALUE_MATE 32000
#define TB_VALUE_INFINITE 32767
#define TB_VALUE_DRAW 0
#define TB_MAX_MATE_PLY 255

/***************************************************************************/
/* ENGINE INTEGRATION CONFIG                                               */
/***************************************************************************/

/*
 * You can override attack generation with engine functions by defining
 * TB_KING_ATTACKS, TB_KNIGHT_ATTACKS, TB_ROOK_ATTACKS, TB_BISHOP_ATTACKS,
 * TB_QUEEN_ATTACKS, TB_PAWN_ATTACKS.
 */
/* #define TB_KING_ATTACKS(square)        <DEFINITION> */
/* #define TB_KNIGHT_ATTACKS(square)      <DEFINITION> */
/* #define TB_ROOK_ATTACKS(square, occ)   <DEFINITION> */
/* #define TB_BISHOP_ATTACKS(square, occ) <DEFINITION> */
/* #define TB_QUEEN_ATTACKS(square, occ)  <DEFINITION> */
/* #define TB_PAWN_ATTACKS(square, color) <DEFINITION> */

#endif
