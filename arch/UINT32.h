/* ******************************************** *\
 *
 *
 *  DATATYPE: the base type of every value.
 *  SDATATYPE: the signed version of DATATYPE.
\* ******************************************** */


/* Including headers */
#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#ifndef STD
#define STD
#endif

#ifndef BITS_PER_REG
#define BITS_PER_REG 32
#endif
#ifndef LOG2_BITS_PER_REG
#define LOG2_BITS_PER_REG 5
#endif

/* Defining 0 and 1 */
#define ZERO 0
#define ONES -1

/* Defining macros */
#define REG_SIZE BITS_PER_REG
#define CHUNK_SIZE 32

#define AND(a,b)  ((a) & (b))
#define OR(a,b)   ((a) | (b))
#define XOR(a,b)  ((a) ^ (b))
#define ANDN(a,b) (~(a) & (b))
#define NOT(a)    (~(a))

/* #define ADD(a,b,c) ((a) + (b)) */
/* #define SUB(a,b,c) ((a) - (b)) */

#define MUL_SIGNED(a,b,c) a * b
#define ADD_SIGNED(a,b,c) a + b
#define SUB_SIGNED(a,b,c) a - b

#define FUNC_AND  std::bit_and<uint32_t>()
#define FUNC_OR   std::bit_or<uint32_t>()
#define FUNC_XOR  std::bit_xor<uint32_t>()
#define FUNC_NOT  std::bit_not<uint32_t>()
#define FUNC_ADD32 std::plus<uint32_t>()
#define FUNC_SUB32 std::minus<uint32_t>()
#define FUNC_MUL32 std::multiplies<uint32_t>()
#define FUNC_ADD64 std::plus<uint32_t>()
#define FUNC_SUB64 std::minus<uint32_t>()
#define FUNC_MUL64 std::multiplies<uint32_t>()


#define ROTATE_MASK(x) (x == 64 ? -1ULL : x == 32 ? -1 : x == 16 ? 0xFFFF : \
    ({ fprintf(stderr,"Not implemented rotate [uint%d_t]. Exiting.\n",x); \
      exit(1); 1; }))

#define L_SHIFT(a,b,c) (c == 4 ? ((a) << (b)) & 0xf : ((a) << (b)))
#define R_SHIFT(a,b,c) ((a) >> (b))
#define RA_SHIFT(a,b,c) (((SDATATYPE)(a)) >> (b))
#define L_ROTATE(a,b,c) ((a << b) | ((a&ROTATE_MASK(c)) >> (c-b)))
#define R_ROTATE(a,b,c) (((a&ROTATE_MASK(c)) >> b) | (a << (c-b)))

#define LIFT_4(x)  (x)
#define LIFT_8(x)  (x)
#define LIFT_16(x) (x)
#define LIFT_32(x) (x)
#define LIFT_64(x) (x)

#define BITMASK(x,n,c) -(((x) >> (n)) & 1)

#define PACK_8x2_to_16(a,b)  ((((uint16_t)(a)) << 8) | ((uint16_t) (b)))
#define PACK_16x2_to_32(a,b) ((((uint32_t)(a)) << 16) | ((uint32_t) (b)))
#define PACK_32x2_to_64(a,b) ((((uint64_t)(a)) << 32) | ((uint64_t) (b)))


#define refresh(x,y) *(y) = x

#ifndef DATATYPE
#if BITS_PER_REG == 4
#define DATATYPE uint8_t // TODO: use something else? do something else?
                         // (needed for Photon right now)
#define SDATATYPE int8_t
#elif BITS_PER_REG == 8
#define DATATYPE uint8_t
#define SDATATYPE int8_t
#elif BITS_PER_REG == 16
#define DATATYPE uint16_t
#define SDATATYPE int16_t
#elif BITS_PER_REG == 32
#define DATATYPE uint32_t
#define SDATATYPE int32_t
#else
#define DATATYPE uint64_t
#define SDATATYPE int64_t
#endif
#endif

#define SET_ALL_ONE()  -1
#define SET_ALL_ZERO() 0

#define ORTHOGONALIZE(in,out)   orthogonalize(in,out)
#define UNORTHOGONALIZE(in,out) unorthogonalize(in,out)

#define ALLOC(size) malloc(size * sizeof(uint64_t))
/* #define NEW(var) (sizeof(var) > 0) ? new var : NULL */ 
#define NEW(var) new var 


#ifdef RUNTIME


/* Orthogonalization stuffs */
static uint64_t mask_l[6] = {
	0xaaaaaaaaaaaaaaaaUL,
	0xccccccccccccccccUL,
	0xf0f0f0f0f0f0f0f0UL,
	0xff00ff00ff00ff00UL,
	0xffff0000ffff0000UL,
	0xffffffff00000000UL
};

static uint64_t mask_r[6] = {
	0x5555555555555555UL,
	0x3333333333333333UL,
	0x0f0f0f0f0f0f0f0fUL,
	0x00ff00ff00ff00ffUL,
	0x0000ffff0000ffffUL,
	0x00000000ffffffffUL
};


void real_ortho(uint64_t data[]) {
  for (int i = 0; i < 5; i ++) {
    int nu = (1UL << i);
    for (int j = 0; j < 32; j += (2 * nu))
      for (int k = 0; k < nu; k ++) {
        uint64_t u = data[j + k] & mask_l[i];
        uint64_t v = data[j + k] & mask_r[i];
        uint64_t x = data[j + nu + k] & mask_l[i];
        uint64_t y = data[j + nu + k] & mask_r[i];
        data[j + k] = u | (x >> nu);
        data[j + nu + k] = (v << nu) | y;
      }
  }
}

#ifdef ORTHO

void orthogonalize(uint64_t* data, uint32_t* out) {
  real_ortho(data);
  for (int i = 0; i < 64; i++)
    out[i] = ((uint32_t*) data)[i];
  
}

void unorthogonalize(uint32_t *in, uint64_t* data) {
  for (int i = 0; i < 32; i++)
    data[i] = ((uint64_t*)in)[i];
  real_ortho(data);
}



#else

void orthogonalize(uint32_t* data, uint32_t* out) {
  for (int i = 0; i < 32; i++)
    out[i] = data[i];
}

void unorthogonalize(uint32_t *in, uint32_t* data) {
  for (int i = 0; i < 32; i++)
    data[i] = in[i];
}


#endif /* ORTHO */

#endif /* NO_RUNTIME */
