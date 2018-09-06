/* Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )

   Based on original fortran 77 code from FFTPACKv4 from NETLIB
   (http://www.netlib.org/fftpack), authored by Dr Paul Swarztrauber
   of NCAR, in 1985.

   As confirmed by the NCAR fftpack software curators, the following
   FFTPACKv5 license applies to FFTPACKv4 sources. My changes are
   released under the same terms.

   FFTPACK license:

   http://www.cisl.ucar.edu/css/software/fftpack5/ftpk.html

   Copyright (c) 2004 the University Corporation for Atmospheric
   Research ("UCAR"). All rights reserved. Developed by NCAR's
   Computational and Information Systems Laboratory, UCAR,
   www.cisl.ucar.edu.

   Redistribution and use of the Software in source and binary forms,
   with or without modification, is permitted provided that the
   following conditions are met:

   - Neither the names of NCAR's Computational and Information Systems
   Laboratory, the University Corporation for Atmospheric Research,
   nor the names of its sponsors or contributors may be used to
   endorse or promote products derived from this Software without
   specific prior written permission.

   - Redistributions of source code must retain the above copyright
   notices, this list of conditions, and the disclaimer below.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions, and the disclaimer below in the
   documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
   HOLDERS BE LIABLE FOR ANY CLAIM, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
   SOFTWARE.


   PFFFT : a Pretty Fast FFT.

   This file is largerly based on the original FFTPACK implementation, modified in
   order to take advantage of SIMD instructions of modern CPUs.
*/

/*
  ChangeLog:
  - 2011/10/02, version 1: This is the very first release of this file.
    2018/09/06,  version 1+: make code easy readable, add Bluestein fft implementation and simple usage function
*/

#include "pffft.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#ifndef M_SQRT2
#define M_SQRT2    1.41421356237309504880   // sqrt(2)
#endif
#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

/* detect compiler flavour */
#if defined(_MSC_VER)
#  define COMPILER_MSVC
#elif defined(__GNUC__)
#  define COMPILER_GCC
#endif

#if defined(COMPILER_GCC)
#  define ALWAYS_INLINE(return_type) inline return_type __attribute__ ((always_inline))
#  define NEVER_INLINE(return_type) return_type __attribute__ ((noinline))
#  define RESTRICT __restrict
#  define VLA_ARRAY_ON_STACK(type__, varname__, size__) type__ varname__[size__];
#elif defined(COMPILER_MSVC)
#  define ALWAYS_INLINE(return_type) __forceinline return_type
#  define NEVER_INLINE(return_type) __declspec(noinline) return_type
#  define RESTRICT __restrict
#  define VLA_ARRAY_ON_STACK(type__, varname__, size__) type__ *(varname__) = (type__*)_alloca((size__) * sizeof(type__))
#endif

/*
   vector support macros: the rest of the code is independant of
   SSE/Altivec/NEON -- adding support for other platforms with 4-element
   vectors should be limited to these macros
*/

// define PFFFT_SIMD_DISABLE if you want to use scalar code instead of simd code
//#define PFFFT_SIMD_DISABLE

/*
   Altivec support macros
*/
#if !defined(PFFFT_SIMD_DISABLE) && (defined(__ppc__) || defined(__ppc64__))
typedef vector float v4sf;
#  define SIMD_SZ 4
#  define VZERO() ((vector float) vec_splat_u8(0))
#  define VMUL(a,b) vec_madd(a,b, VZERO())
#  define VADD(a,b) vec_add(a,b)
#  define VMADD(a,b,c) vec_madd(a,b,c)
#  define VSUB(a,b) vec_sub(a,b)
inline v4sf ld_ps1(const float *p) { v4sf v = vec_lde(0, p); return vec_splat(vec_perm(v, v, vec_lvsl(0, p)), 0); }
#  define LD_PS1(p) ld_ps1(&p)
#  define INTERLEAVE2(in1, in2, out1, out2) { v4sf tmp__ = vec_mergeh(in1, in2); out2 = vec_mergel(in1, in2); out1 = tmp__; }
#  define UNINTERLEAVE2(in1, in2, out1, out2) {                           \
    vector unsigned char vperm1 =  (vector unsigned char)(0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27); \
    vector unsigned char vperm2 =  (vector unsigned char)(4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31); \
    v4sf tmp__ = vec_perm(in1, in2, vperm1); out2 = vec_perm(in1, in2, vperm2); out1 = tmp__; \
  }
#  define VTRANSPOSE4(x0,x1,x2,x3) {              \
    v4sf y0 = vec_mergeh(x0, x2);               \
    v4sf y1 = vec_mergel(x0, x2);               \
    v4sf y2 = vec_mergeh(x1, x3);               \
    v4sf y3 = vec_mergel(x1, x3);               \
    x0 = vec_mergeh(y0, y2);                    \
    x1 = vec_mergel(y0, y2);                    \
    x2 = vec_mergeh(y1, y3);                    \
    x3 = vec_mergel(y1, y3);                    \
  }
#  define VSWAPHL(a,b) vec_perm(a,b, (vector unsigned char)(16,17,18,19,20,21,22,23,8,9,10,11,12,13,14,15))
#  define VALIGNED(ptr) ((((long)(ptr)) & 0xF) == 0)

/*
  SSE1 support macros
*/
#elif !defined(PFFFT_SIMD_DISABLE) && (defined(__x86_64__) || defined(_M_X64) || defined(i386) || defined(_M_IX86))

#include <xmmintrin.h>

typedef __m128 v4sf;
#  define SIMD_SZ 4 // 4 floats by simd vector -- this is pretty much hardcoded in the preprocess/finalize functions anyway so you will have to work if you want to enable AVX with its 256-bit vectors.
#  define VZERO() _mm_setzero_ps()
#  define VMUL(a, b) _mm_mul_ps(a,b)
#  define VADD(a, b) _mm_add_ps(a,b)
#  define VMADD(a, b, c) _mm_add_ps(_mm_mul_ps(a,b), c)
#  define VSUB(a, b) _mm_sub_ps(a,b)
#  define LD_PS1(p) _mm_set1_ps(p)
#  define INTERLEAVE2(in1, in2, out1, out2) { v4sf tmp__ = _mm_unpacklo_ps(in1, in2); (out2) = _mm_unpackhi_ps(in1, in2); (out1) = tmp__; }
#  define UNINTERLEAVE2(in1, in2, out1, out2) { v4sf tmp__ = _mm_shuffle_ps(in1, in2, _MM_SHUFFLE(2,0,2,0)); (out2) = _mm_shuffle_ps(in1, in2, _MM_SHUFFLE(3,1,3,1)); (out1) = tmp__; }
#  define VTRANSPOSE4(x0, x1, x2, x3) _MM_TRANSPOSE4_PS(x0,x1,x2,x3)
#  define VSWAPHL(a, b) _mm_shuffle_ps(b, a, _MM_SHUFFLE(3,2,1,0))
#  define VALIGNED(ptr) ((((long)(ptr)) & 0xF) == 0)

/*
  ARM NEON support macros
*/
#elif !defined(PFFFT_SIMD_DISABLE) && (defined(__arm__) || defined(__aarch64__) || defined(__arm64__))
#  include <arm_neon.h>
typedef float32x4_t v4sf;
#  define SIMD_SZ 4
#  define VZERO() vdupq_n_f32(0)
#  define VMUL(a,b) vmulq_f32(a,b)
#  define VADD(a,b) vaddq_f32(a,b)
#  define VMADD(a,b,c) vmlaq_f32(c,a,b)
#  define VSUB(a,b) vsubq_f32(a,b)
#  define LD_PS1(p) vld1q_dup_f32(&(p))
#  define INTERLEAVE2(in1, in2, out1, out2) { float32x4x2_t tmp__ = vzipq_f32(in1,in2); out1=tmp__.val[0]; out2=tmp__.val[1]; }
#  define UNINTERLEAVE2(in1, in2, out1, out2) { float32x4x2_t tmp__ = vuzpq_f32(in1,in2); out1=tmp__.val[0]; out2=tmp__.val[1]; }
#  define VTRANSPOSE4(x0,x1,x2,x3) {                                    \
    float32x4x2_t t0_ = vzipq_f32(x0, x2);                              \
    float32x4x2_t t1_ = vzipq_f32(x1, x3);                              \
    float32x4x2_t u0_ = vzipq_f32(t0_.val[0], t1_.val[0]);              \
    float32x4x2_t u1_ = vzipq_f32(t0_.val[1], t1_.val[1]);              \
    x0 = u0_.val[0]; x1 = u0_.val[1]; x2 = u1_.val[0]; x3 = u1_.val[1]; \
  }
// marginally faster version
//#  define VTRANSPOSE4(x0,x1,x2,x3) { asm("vtrn.32 %q0, %q1;\n vtrn.32 %q2,%q3\n vswp %f0,%e2\n vswp %f1,%e3" : "+w"(x0), "+w"(x1), "+w"(x2), "+w"(x3)::); }
#  define VSWAPHL(a,b) vcombine_f32(vget_low_f32(b), vget_high_f32(a))
#  define VALIGNED(ptr) ((((long)(ptr)) & 0x3) == 0)
#else
#  if !defined(PFFFT_SIMD_DISABLE)
#    warning "building with simd disabled !\n";
#    define PFFFT_SIMD_DISABLE // fallback to scalar code
#  endif
#endif

// fallback mode for situations where SSE/Altivec are not available, use scalar mode instead
#ifdef PFFFT_SIMD_DISABLE
typedef float v4sf;
#  define SIMD_SZ 1
#  define VZERO() 0.f
#  define VMUL(a, b) ((a)*(b))
#  define VADD(a, b) ((a)+(b))
#  define VMADD(a, b, c) ((a)*(b)+(c))
#  define VSUB(a, b) ((a)-(b))
#  define LD_PS1(p) (p)
#  define VALIGNED(ptr) ((((long)(ptr)) & 0x3) == 0)
#endif

// shortcuts for complex multiplcations
#define VCPLXMUL(ar, ai, br, bi) { v4sf tmp; tmp=VMUL(ar,bi); (ar)=VMUL(ar,br); (ar)=VSUB(ar,VMUL(ai,bi)); (ai)=VMUL(ai,br); (ai)=VADD(ai,tmp); }
#define VCPLXMULCONJ(ar, ai, br, bi) { v4sf tmp; tmp=VMUL(ar,bi); (ar)=VMUL(ar,br); (ar)=VADD(ar,VMUL(ai,bi)); (ai)=VMUL(ai,br); (ai)=VSUB(ai,tmp); }
#ifndef SVMUL
// multiply a scalar with a vector
#define SVMUL(f, v) VMUL(LD_PS1(f),v)
#endif

#if !defined(PFFFT_SIMD_DISABLE)
typedef union v4sf_union {
    v4sf v;
    float f[4];
} v4sf_union;

#include <string.h>

#define assertv4(v, f0, f1, f2, f3) assert((v).f[0] == (f0) && (v).f[1] == (f1) && (v).f[2] == (f2) && (v).f[3] == (f3))

/* detect bugs with the vector support macros */
void validate_pffft_simd() {
    float f[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    v4sf_union a0 = {0};
    v4sf_union a1 = {0};
    v4sf_union a2 = {0};
    v4sf_union a3 = {0};
    v4sf_union t = {0};
    v4sf_union u = {0};
    memcpy(a0.f, f, 4 * sizeof(float));
    memcpy(a1.f, f + 4, 4 * sizeof(float));
    memcpy(a2.f, f + 8, 4 * sizeof(float));
    memcpy(a3.f, f + 12, 4 * sizeof(float));

    t = a0;
    u = a1;
    t.v = VZERO();
    printf("VZERO=[%2g %2g %2g %2g]\n", t.f[0], t.f[1], t.f[2], t.f[3]);
    assertv4(t, 0, 0, 0, 0);
    t.v = VADD(a1.v, a2.v);
    printf("VADD(4:7,8:11)=[%2g %2g %2g %2g]\n", t.f[0], t.f[1], t.f[2], t.f[3]);
    assertv4(t, 12, 14, 16, 18);
    t.v = VMUL(a1.v, a2.v);
    printf("VMUL(4:7,8:11)=[%2g %2g %2g %2g]\n", t.f[0], t.f[1], t.f[2], t.f[3]);
    assertv4(t, 32, 45, 60, 77);
    t.v = VMADD(a1.v, a2.v, a0.v);
    printf("VMADD(4:7,8:11,0:3)=[%2g %2g %2g %2g]\n", t.f[0], t.f[1], t.f[2], t.f[3]);
    assertv4(t, 32, 46, 62, 80);

    INTERLEAVE2(a1.v, a2.v, t.v, u.v);
    printf("INTERLEAVE2(4:7,8:11)=[%2g %2g %2g %2g] [%2g %2g %2g %2g]\n", t.f[0], t.f[1], t.f[2], t.f[3], u.f[0],
           u.f[1], u.f[2], u.f[3]);
    assertv4(t, 4, 8, 5, 9);
    assertv4(u, 6, 10, 7, 11);
    UNINTERLEAVE2(a1.v, a2.v, t.v, u.v);
    printf("UNINTERLEAVE2(4:7,8:11)=[%2g %2g %2g %2g] [%2g %2g %2g %2g]\n", t.f[0], t.f[1], t.f[2], t.f[3], u.f[0],
           u.f[1], u.f[2], u.f[3]);
    assertv4(t, 4, 6, 8, 10);
    assertv4(u, 5, 7, 9, 11);

    t.v = LD_PS1(f[15]);
    printf("LD_PS1(15)=[%2g %2g %2g %2g]\n", t.f[0], t.f[1], t.f[2], t.f[3]);
    assertv4(t, 15, 15, 15, 15);
    t.v = VSWAPHL(a1.v, a2.v);
    printf("VSWAPHL(4:7,8:11)=[%2g %2g %2g %2g]\n", t.f[0], t.f[1], t.f[2], t.f[3]);
    assertv4(t, 8, 9, 6, 7);
    VTRANSPOSE4(a0.v, a1.v, a2.v, a3.v);
    printf("VTRANSPOSE4(0:3,4:7,8:11,12:15)=[%2g %2g %2g %2g] [%2g %2g %2g %2g] [%2g %2g %2g %2g] [%2g %2g %2g %2g]\n",
           a0.f[0], a0.f[1], a0.f[2], a0.f[3], a1.f[0], a1.f[1], a1.f[2], a1.f[3],
           a2.f[0], a2.f[1], a2.f[2], a2.f[3], a3.f[0], a3.f[1], a3.f[2], a3.f[3]);
    assertv4(a0, 0, 4, 8, 12);
    assertv4(a1, 1, 5, 9, 13);
    assertv4(a2, 2, 6, 10, 14);
    assertv4(a3, 3, 7, 11, 15);
}

#endif //!PFFFT_SIMD_DISABLE

void *pffft_aligned_calloc(size_t nb_bytes, size_t sizeOfelements) {
    /* SSE and co like 16-bytes aligned pointers */
    const int v4SF_alignment = 64; // with a 64-byte alignment, we are even aligned on L2 cache lines...
    void *p, *p0 = calloc(nb_bytes + (v4SF_alignment / sizeOfelements + 1), sizeOfelements);
    if (!p0) return (void *) 0;
    p = (void *) (((size_t) p0 + v4SF_alignment) & (~((size_t) (v4SF_alignment - 1))));
    *((void **) p - 1) = p0;
    return p;
}

void pffft_aligned_free(void *p) {
    if (p) free(*((void **) p - 1));
}

int pffft_simd_size() { return SIMD_SZ; }


void SinCos(double x, float *pSin, float *pCos) {
    *pCos = (float)cos(x);
    *pSin =  (float)sin(x);
}

/*
  passf2 and passb2 has been merged here, fsign = -1 for passf2, +1 for passb2
*/
static NEVER_INLINE(void) cdft_2(int count, int w_count, const v4sf *input, v4sf *output, const float *twiddle_1,
                                 int isign) {
    int n, i;
    int n_count = w_count * count;
    if (count <= 2) {
        for (n = 0; n < n_count; n += count, output += count, input += 2 * count) {
            output[0] = VADD(input[0], input[count + 0]);
            output[n_count] = VSUB(input[0], input[count + 0]);
            output[1] = VADD(input[1], input[count + 1]);
            output[n_count + 1] = VSUB(input[1], input[count + 1]);
        }
    } else {
        for (n = 0; n < n_count; n += count, output += count, input += 2 * count) {
            for (i = 0; i < count - 1; i += 2) {
                v4sf tr2 = VSUB(input[i + 0], input[i + count + 0]);
                v4sf ti2 = VSUB(input[i + 1], input[i + count + 1]);
                v4sf wr = LD_PS1(twiddle_1[i]), wi = VMUL(LD_PS1(isign), LD_PS1(twiddle_1[i + 1]));
                output[i] = VADD(input[i + 0], input[i + count + 0]);
                output[i + 1] = VADD(input[i + 1], input[i + count + 1]);
                VCPLXMUL(tr2, ti2, wr, wi);
                output[i + n_count] = tr2;
                output[i + n_count + 1] = ti2;
            }
        }
    }
}

/*
  passf3 and passb3 has been merged here, fsign = -1 for passf3, +1 for passb3
*/
static NEVER_INLINE(void) cdft_3(int count, int w_count, const v4sf *input, v4sf *output,
                                 const float *twiddle_1, const float *twiddle_2, int isign) {
    static const float taur = -0.5f;
    float taui = 0.866025403784439f * isign;
    int i, n;
    v4sf tr2, ti2, cr2, ci2, cr3, ci3, dr2, di2, dr3, di3;
    int n_count = w_count * count;
    float wr1, wi1, wr2, wi2;
    assert(count > 2);
    for (n = 0; n < n_count; n += count, input += 3 * count, output += count) {
        for (i = 0; i < count - 1; i += 2) {
            tr2 = VADD(input[i + count], input[i + 2 * count]);
            cr2 = VADD(input[i], SVMUL(taur, tr2));
            output[i] = VADD(input[i], tr2);
            ti2 = VADD(input[i + count + 1], input[i + 2 * count + 1]);
            ci2 = VADD(input[i + 1], SVMUL(taur, ti2));
            output[i + 1] = VADD(input[i + 1], ti2);
            cr3 = SVMUL(taui, VSUB(input[i + count], input[i + 2 * count]));
            ci3 = SVMUL(taui, VSUB(input[i + count + 1], input[i + 2 * count + 1]));
            dr2 = VSUB(cr2, ci3);
            dr3 = VADD(cr2, ci3);
            di2 = VADD(ci2, cr3);
            di3 = VSUB(ci2, cr3);
            wr1 = twiddle_1[i], wi1 = isign * twiddle_1[i + 1], wr2 = twiddle_2[i], wi2 = isign * twiddle_2[i + 1];
            VCPLXMUL(dr2, di2, LD_PS1(wr1), LD_PS1(wi1));
            output[i + n_count] = dr2;
            output[i + n_count + 1] = di2;
            VCPLXMUL(dr3, di3, LD_PS1(wr2), LD_PS1(wi2));
            output[i + 2 * n_count] = dr3;
            output[i + 2 * n_count + 1] = di3;
        }
    }
} /* passf3 */

static NEVER_INLINE(void) cdft_4(int count, int w_count, const v4sf *input, v4sf *output,
                                 const float *twiddle_1, const float *twiddle_2, const float *twiddle_3, int isign) {
    /* isign == -1 for forward transform and +1 for backward transform */

    int i, n;
    v4sf ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
    int n_count = w_count * count;
    if (count == 2) {
        for (n = 0; n < n_count; n += count, output += count, input += 4 * count) {
            tr1 = VSUB(input[0], input[2 * count + 0]);
            tr2 = VADD(input[0], input[2 * count + 0]);
            ti1 = VSUB(input[1], input[2 * count + 1]);
            ti2 = VADD(input[1], input[2 * count + 1]);
            ti4 = VMUL(VSUB(input[1 * count + 0], input[3 * count + 0]), LD_PS1(isign));
            tr4 = VMUL(VSUB(input[3 * count + 1], input[1 * count + 1]), LD_PS1(isign));
            tr3 = VADD(input[count + 0], input[3 * count + 0]);
            ti3 = VADD(input[count + 1], input[3 * count + 1]);

            output[0 * n_count + 0] = VADD(tr2, tr3);
            output[0 * n_count + 1] = VADD(ti2, ti3);
            output[1 * n_count + 0] = VADD(tr1, tr4);
            output[1 * n_count + 1] = VADD(ti1, ti4);
            output[2 * n_count + 0] = VSUB(tr2, tr3);
            output[2 * n_count + 1] = VSUB(ti2, ti3);
            output[3 * n_count + 0] = VSUB(tr1, tr4);
            output[3 * n_count + 1] = VSUB(ti1, ti4);
        }
    } else {
        for (n = 0; n < n_count; n += count, output += count, input += 4 * count) {
            for (i = 0; i < count - 1; i += 2) {
                float wr1, wi1, wr2, wi2, wr3, wi3;
                tr1 = VSUB(input[i + 0], input[i + 2 * count + 0]);
                tr2 = VADD(input[i + 0], input[i + 2 * count + 0]);
                ti1 = VSUB(input[i + 1], input[i + 2 * count + 1]);
                ti2 = VADD(input[i + 1], input[i + 2 * count + 1]);
                tr4 = VMUL(VSUB(input[i + 3 * count + 1], input[i + 1 * count + 1]), LD_PS1(isign));
                ti4 = VMUL(VSUB(input[i + 1 * count + 0], input[i + 3 * count + 0]), LD_PS1(isign));
                tr3 = VADD(input[i + count + 0], input[i + 3 * count + 0]);
                ti3 = VADD(input[i + count + 1], input[i + 3 * count + 1]);

                output[i] = VADD(tr2, tr3);
                cr3 = VSUB(tr2, tr3);
                output[i + 1] = VADD(ti2, ti3);
                ci3 = VSUB(ti2, ti3);

                cr2 = VADD(tr1, tr4);
                cr4 = VSUB(tr1, tr4);
                ci2 = VADD(ti1, ti4);
                ci4 = VSUB(ti1, ti4);
                wr1 = twiddle_1[i], wi1 = isign * twiddle_1[i + 1];
                VCPLXMUL(cr2, ci2, LD_PS1(wr1), LD_PS1(wi1));
                wr2 = twiddle_2[i], wi2 = isign * twiddle_2[i + 1];
                output[i + n_count] = cr2;
                output[i + n_count + 1] = ci2;

                VCPLXMUL(cr3, ci3, LD_PS1(wr2), LD_PS1(wi2));
                wr3 = twiddle_3[i], wi3 = isign * twiddle_3[i + 1];
                output[i + 2 * n_count] = cr3;
                output[i + 2 * n_count + 1] = ci3;

                VCPLXMUL(cr4, ci4, LD_PS1(wr3), LD_PS1(wi3));
                output[i + 3 * n_count] = cr4;
                output[i + 3 * n_count + 1] = ci4;
            }
        }
    }
} /* passf4 */

/*
  passf5 and passb5 has been merged here, fsign = -1 for passf5, +1 for passb5
*/
static NEVER_INLINE(void) cdft_5(int count, int w_count, const v4sf *input, v4sf *output,
                                 const float *twiddle_1, const float *twiddle_2,
                                 const float *twiddle_3, const float *twiddle_4, int isign) {
    static const float tr11 = .309016994374947f;
    const float ti11 = .951056516295154f * isign;
    static const float tr12 = -.809016994374947f;
    const float ti12 = .587785252292473f * isign;

    /* Local variables */
    int i, n;
    v4sf ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, cr4, ti2, ti3,
            ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3, tr4, tr5;

    float wr1, wi1, wr2, wi2, wr3, wi3, wr4, wi4;

#define c5_input_ref(a_1, a_2) input[((a_2)-1)*count + (a_1) + 1]
#define c5_output_ref(a_1, a_3) output[((a_3)-1)*w_count*count + (a_1) + 1]

    assert(count > 2);
    for (n = 0; n < w_count; ++n, input += 5 * count, output += count) {
        for (i = 0; i < count - 1; i += 2) {
            ti5 = VSUB(c5_input_ref(i, 2), c5_input_ref(i, 5));
            ti2 = VADD(c5_input_ref(i, 2), c5_input_ref(i, 5));
            ti4 = VSUB(c5_input_ref(i, 3), c5_input_ref(i, 4));
            ti3 = VADD(c5_input_ref(i, 3), c5_input_ref(i, 4));
            tr5 = VSUB(c5_input_ref(i - 1, 2), c5_input_ref(i - 1, 5));
            tr2 = VADD(c5_input_ref(i - 1, 2), c5_input_ref(i - 1, 5));
            tr4 = VSUB(c5_input_ref(i - 1, 3), c5_input_ref(i - 1, 4));
            tr3 = VADD(c5_input_ref(i - 1, 3), c5_input_ref(i - 1, 4));
            c5_output_ref(i - 1, 1) = VADD(c5_input_ref(i - 1, 1), VADD(tr2, tr3));
            c5_output_ref(i, 1) = VADD(c5_input_ref(i, 1), VADD(ti2, ti3));
            cr2 = VADD(c5_input_ref(i - 1, 1), VADD(SVMUL(tr11, tr2), SVMUL(tr12, tr3)));
            ci2 = VADD(c5_input_ref(i, 1), VADD(SVMUL(tr11, ti2), SVMUL(tr12, ti3)));
            cr3 = VADD(c5_input_ref(i - 1, 1), VADD(SVMUL(tr12, tr2), SVMUL(tr11, tr3)));
            ci3 = VADD(c5_input_ref(i, 1), VADD(SVMUL(tr12, ti2), SVMUL(tr11, ti3)));
            cr5 = VADD(SVMUL(ti11, tr5), SVMUL(ti12, tr4));
            ci5 = VADD(SVMUL(ti11, ti5), SVMUL(ti12, ti4));
            cr4 = VSUB(SVMUL(ti12, tr5), SVMUL(ti11, tr4));
            ci4 = VSUB(SVMUL(ti12, ti5), SVMUL(ti11, ti4));
            dr3 = VSUB(cr3, ci4);
            dr4 = VADD(cr3, ci4);
            di3 = VADD(ci3, cr4);
            di4 = VSUB(ci3, cr4);
            dr5 = VADD(cr2, ci5);
            dr2 = VSUB(cr2, ci5);
            di5 = VSUB(ci2, cr5);
            di2 = VADD(ci2, cr5);
            wr1 = twiddle_1[i], wi1 = isign * twiddle_1[i + 1], wr2 = twiddle_2[i], wi2 = isign * twiddle_2[i + 1];
            wr3 = twiddle_3[i], wi3 = isign * twiddle_3[i + 1], wr4 = twiddle_4[i], wi4 = isign * twiddle_4[i + 1];
            VCPLXMUL(dr2, di2, LD_PS1(wr1), LD_PS1(wi1));
            c5_output_ref(i - 1, 2) = dr2;
            c5_output_ref(i, 2) = di2;
            VCPLXMUL(dr3, di3, LD_PS1(wr2), LD_PS1(wi2));
            c5_output_ref(i - 1, 3) = dr3;
            c5_output_ref(i, 3) = di3;
            VCPLXMUL(dr4, di4, LD_PS1(wr3), LD_PS1(wi3));
            c5_output_ref(i - 1, 4) = dr4;
            c5_output_ref(i, 4) = di4;
            VCPLXMUL(dr5, di5, LD_PS1(wr4), LD_PS1(wi4));
            c5_output_ref(i - 1, 5) = dr5;
            c5_output_ref(i, 5) = di5;
        }
    }
#undef c5_output_ref
#undef c5_input_ref
}

static NEVER_INLINE(void) dft_2_r(int count, int w_count, const v4sf *RESTRICT input, v4sf *RESTRICT output,
                                  const float *twiddle_1) {
    static const float minus_one = -1.f;
    int i, n, n_count = w_count * count;
    for (n = 0; n < n_count; n += count) {
        v4sf a = input[n], b = input[n + n_count];
        output[2 * n] = VADD(a, b);
        output[2 * (n + count) - 1] = VSUB(a, b);
    }
    if (count < 2) return;
    if (count != 2) {
        for (n = 0; n < n_count; n += count) {
            for (i = 2; i < count; i += 2) {
                v4sf tr2 = input[i - 1 + n + n_count], ti2 = input[i + n + n_count];
                v4sf br = input[i - 1 + n], bi = input[i + n];
                VCPLXMULCONJ(tr2, ti2, LD_PS1(twiddle_1[i - 2]), LD_PS1(twiddle_1[i - 1]));
                output[i + 2 * n] = VADD(bi, ti2);
                output[2 * (n + count) - i] = VSUB(ti2, bi);
                output[i - 1 + 2 * n] = VADD(br, tr2);
                output[2 * (n + count) - i - 1] = VSUB(br, tr2);
            }
        }
        if (count % 2 == 1) return;
    }
    for (n = 0; n < n_count; n += count) {
        output[2 * n + count] = SVMUL(minus_one, input[count - 1 + n + n_count]);
        output[2 * n + count - 1] = input[n + count - 1];
    }
} /* radf2 */

static NEVER_INLINE(void) idft_2_r(int count, int w_count, const v4sf *input, v4sf *output, const float *twiddle_1) {
    static const float minus_two = -2;
    int i, n, n_count = w_count * count;
    v4sf a, b, c, d, tr2, ti2;
    for (n = 0; n < n_count; n += count) {
        a = input[2 * n];
        b = input[2 * (n + count) - 1];
        output[n] = VADD(a, b);
        output[n + n_count] = VSUB(a, b);
    }
    if (count < 2) return;
    if (count != 2) {
        for (n = 0; n < n_count; n += count) {
            for (i = 2; i < count; i += 2) {
                a = input[i - 1 + 2 * n];
                b = input[2 * (n + count) - i - 1];
                c = input[i + 0 + 2 * n];
                d = input[2 * (n + count) - i + 0];
                output[i - 1 + n] = VADD(a, b);
                tr2 = VSUB(a, b);
                output[i + 0 + n] = VSUB(c, d);
                ti2 = VADD(c, d);
                VCPLXMUL(tr2, ti2, LD_PS1(twiddle_1[i - 2]), LD_PS1(twiddle_1[i - 1]));
                output[i - 1 + n + n_count] = tr2;
                output[i + 0 + n + n_count] = ti2;
            }
        }
        if (count % 2 == 1) return;
    }
    for (n = 0; n < n_count; n += count) {
        a = input[2 * n + count - 1];
        b = input[2 * n + count];
        output[n + count - 1] = VADD(a, a);
        output[n + count - 1 + n_count] = SVMUL(minus_two, b);
    }
} /* radb2 */

static void dft_3_r(int count, int w_count, const v4sf *RESTRICT input, v4sf *RESTRICT output,
                    const float *twiddle_1, const float *twiddle_2) {
    static const float taur = -0.5f;
    static const float taui = 0.866025403784439f;
    int i, n, j;
    v4sf ci2, di2, di3, cr2, dr2, dr3, ti2, ti3, tr2, tr3, wr1, wi1, wr2, wi2;
    for (n = 0; n < w_count; n++) {
        cr2 = VADD(input[(n + w_count) * count], input[(n + 2 * w_count) * count]);
        output[3 * n * count] = VADD(input[n * count], cr2);
        output[(3 * n + 2) * count] = SVMUL(taui, VSUB(input[(n + w_count * 2) * count], input[(n + w_count) * count]));
        output[count - 1 + (3 * n + 1) * count] = VADD(input[n * count], SVMUL(taur, cr2));
    }
    if (count == 1) return;
    for (n = 0; n < w_count; n++) {
        for (i = 2; i < count; i += 2) {
            j = count - i;
            wr1 = LD_PS1(twiddle_1[i - 2]);
            wi1 = LD_PS1(twiddle_1[i - 1]);
            dr2 = input[i - 1 + (n + w_count) * count];
            di2 = input[i + (n + w_count) * count];
            VCPLXMULCONJ(dr2, di2, wr1, wi1);

            wr2 = LD_PS1(twiddle_2[i - 2]);
            wi2 = LD_PS1(twiddle_2[i - 1]);
            dr3 = input[i - 1 + (n + w_count * 2) * count];
            di3 = input[i + (n + w_count * 2) * count];
            VCPLXMULCONJ(dr3, di3, wr2, wi2);

            cr2 = VADD(dr2, dr3);
            ci2 = VADD(di2, di3);
            output[i - 1 + 3 * n * count] = VADD(input[i - 1 + n * count], cr2);
            output[i + 3 * n * count] = VADD(input[i + n * count], ci2);
            tr2 = VADD(input[i - 1 + n * count], SVMUL(taur, cr2));
            ti2 = VADD(input[i + n * count], SVMUL(taur, ci2));
            tr3 = SVMUL(taui, VSUB(di2, di3));
            ti3 = SVMUL(taui, VSUB(dr3, dr2));
            output[i - 1 + (3 * n + 2) * count] = VADD(tr2, tr3);
            output[j - 1 + (3 * n + 1) * count] = VSUB(tr2, tr3);
            output[i + (3 * n + 2) * count] = VADD(ti2, ti3);
            output[j + (3 * n + 1) * count] = VSUB(ti3, ti2);
        }
    }
} /* radf3 */

static void idft_3_r(int count, int w_count, const v4sf *RESTRICT input, v4sf *RESTRICT output,
                     const float *twiddle_1, const float *twiddle_2) {
    static const float taur = -0.5f;
    static const float taui = 0.866025403784439f;
    static const float taui_2 = 0.866025403784439f * 2;
    int i, n, j;
    v4sf ci2, ci3, di2, di3, cr2, cr3, dr2, dr3, ti2, tr2;
    for (n = 0; n < w_count; n++) {
        tr2 = input[count - 1 + (3 * n + 1) * count];
        tr2 = VADD(tr2, tr2);
        cr2 = VMADD(LD_PS1(taur), tr2, input[3 * n * count]);
        output[n * count] = VADD(input[3 * n * count], tr2);
        ci3 = SVMUL(taui_2, input[(3 * n + 2) * count]);
        output[(n + w_count) * count] = VSUB(cr2, ci3);
        output[(n + 2 * w_count) * count] = VADD(cr2, ci3);
    }
    if (count == 1) return;
    for (n = 0; n < w_count; n++) {
        for (i = 2; i < count; i += 2) {
            j = count - i;
            tr2 = VADD(input[i - 1 + (3 * n + 2) * count], input[j - 1 + (3 * n + 1) * count]);
            cr2 = VMADD(LD_PS1(taur), tr2, input[i - 1 + 3 * n * count]);
            output[i - 1 + n * count] = VADD(input[i - 1 + 3 * n * count], tr2);
            ti2 = VSUB(input[i + (3 * n + 2) * count], input[j + (3 * n + 1) * count]);
            ci2 = VMADD(LD_PS1(taur), ti2, input[i + 3 * n * count]);
            output[i + n * count] = VADD(input[i + 3 * n * count], ti2);
            cr3 = SVMUL(taui, VSUB(input[i - 1 + (3 * n + 2) * count], input[j - 1 + (3 * n + 1) * count]));
            ci3 = SVMUL(taui, VADD(input[i + (3 * n + 2) * count], input[j + (3 * n + 1) * count]));
            dr2 = VSUB(cr2, ci3);
            dr3 = VADD(cr2, ci3);
            di2 = VADD(ci2, cr3);
            di3 = VSUB(ci2, cr3);
            VCPLXMUL(dr2, di2, LD_PS1(twiddle_1[i - 2]), LD_PS1(twiddle_1[i - 1]));
            output[i - 1 + (n + w_count) * count] = dr2;
            output[i + (n + w_count) * count] = di2;
            VCPLXMUL(dr3, di3, LD_PS1(twiddle_2[i - 2]), LD_PS1(twiddle_2[i - 1]));
            output[i - 1 + (n + 2 * w_count) * count] = dr3;
            output[i + (n + 2 * w_count) * count] = di3;
        }
    }
} /* radb3 */

static NEVER_INLINE(void) dft_4_r(int count, int w_count, const v4sf *RESTRICT input, v4sf *RESTRICT output,
                                  const float *RESTRICT twiddle_1, const float *RESTRICT twiddle_2,
                                  const float *RESTRICT twiddle_3) {
    static const float minus_hsqt2 = (float) -0.7071067811865475;
    int i, n, n_count = w_count * count;
    {
        const v4sf *RESTRICT in = input, *RESTRICT input_end = input + n_count;
        v4sf *RESTRICT out = output;
        while (input < input_end) {
            // this loop represents between 25% and 40% of total radf4_ps cost !
            v4sf a0 = input[0], a1 = input[n_count];
            v4sf a2 = input[2 * n_count], a3 = input[3 * n_count];
            v4sf tr1 = VADD(a1, a3);
            v4sf tr2 = VADD(a0, a2);
            output[2 * count - 1] = VSUB(a0, a2);
            output[2 * count] = VSUB(a3, a1);
            output[0] = VADD(tr1, tr2);
            output[4 * count - 1] = VSUB(tr2, tr1);
            input += count;
            output += 4 * count;
        }
        input = in;
        output = out;
    }
    if (count < 2) return;
    if (count != 2) {
        for (n = 0; n < n_count; n += count) {
            const v4sf *RESTRICT pre_input = (v4sf *) (input + 1 + n);
            for (i = 2; i < count; i += 2, pre_input += 2) {
                int j = count - i;
                v4sf wr, wi, cr2, ci2, cr3, ci3, cr4, ci4;
                v4sf tr1, ti1, tr2, ti2, tr3, ti3, tr4, ti4;

                cr2 = pre_input[1 * n_count + 0];
                ci2 = pre_input[1 * n_count + 1];
                wr = LD_PS1(twiddle_1[i - 2]);
                wi = LD_PS1(twiddle_1[i - 1]);
                VCPLXMULCONJ(cr2, ci2, wr, wi);

                cr3 = pre_input[2 * n_count + 0];
                ci3 = pre_input[2 * n_count + 1];
                wr = LD_PS1(twiddle_2[i - 2]);
                wi = LD_PS1(twiddle_2[i - 1]);
                VCPLXMULCONJ(cr3, ci3, wr, wi);

                cr4 = pre_input[3 * n_count];
                ci4 = pre_input[3 * n_count + 1];
                wr = LD_PS1(twiddle_3[i - 2]);
                wi = LD_PS1(twiddle_3[i - 1]);
                VCPLXMULCONJ(cr4, ci4, wr, wi);

                /* at this point, on SSE, five of "cr2 cr3 cr4 ci2 ci3 ci4" should be loaded in registers */

                tr1 = VADD(cr2, cr4);
                tr4 = VSUB(cr4, cr2);
                tr2 = VADD(pre_input[0], cr3);
                tr3 = VSUB(pre_input[0], cr3);
                output[i - 1 + 4 * n] = VADD(tr1, tr2);
                output[j - 1 + 4 * n + 3 * count] = VSUB(tr2, tr1); // at this point tr1 and tr2 can be disposed
                ti1 = VADD(ci2, ci4);
                ti4 = VSUB(ci2, ci4);
                output[i - 1 + 4 * n + 2 * count] = VADD(ti4, tr3);
                output[j - 1 + 4 * n + 1 * count] = VSUB(tr3, ti4); // dispose tr3, ti4
                ti2 = VADD(pre_input[1], ci3);
                ti3 = VSUB(pre_input[1], ci3);
                output[i + 4 * n] = VADD(ti1, ti2);
                output[j + 4 * n + 3 * count] = VSUB(ti1, ti2);
                output[i + 4 * n + 2 * count] = VADD(tr4, ti3);
                output[j + 4 * n + 1 * count] = VSUB(tr4, ti3);
            }
        }
        if (count % 2 == 1) return;
    }
    for (n = 0; n < n_count; n += count) {
        v4sf a = input[count - 1 + n + n_count], b = input[count - 1 + n + 3 * n_count];
        v4sf c = input[count - 1 + n], d = input[count - 1 + n + 2 * n_count];
        v4sf ti1 = SVMUL(minus_hsqt2, VADD(a, b));
        v4sf tr1 = SVMUL(minus_hsqt2, VSUB(b, a));
        output[count - 1 + 4 * n] = VADD(tr1, c);
        output[count - 1 + 4 * n + 2 * count] = VSUB(c, tr1);
        output[4 * n + 1 * count] = VSUB(ti1, d);
        output[4 * n + 3 * count] = VADD(ti1, d);
    }
} /* radf4 */

static NEVER_INLINE(void) idft_4_r(int count, int w_count, const v4sf *RESTRICT input, v4sf *RESTRICT output,
                                   const float *RESTRICT twiddle_1, const float *RESTRICT twiddle_2,
                                   const float *RESTRICT twiddle_3) {
    static const float minus_sqrt2 = (float) -1.414213562373095;
    static const float two = 2.f;
    int i, n, n_count = w_count * count;
    v4sf ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
    {
        const v4sf *RESTRICT in = input, *RESTRICT output_end = output + n_count;
        v4sf *out = output;
        while (output < output_end) {
            v4sf a = input[0], b = input[4 * count - 1];
            v4sf c = input[2 * count], d = input[2 * count - 1];
            tr3 = SVMUL(two, d);
            tr2 = VADD(a, b);
            tr1 = VSUB(a, b);
            tr4 = SVMUL(two, c);
            output[0 * n_count] = VADD(tr2, tr3);
            output[2 * n_count] = VSUB(tr2, tr3);
            output[1 * n_count] = VSUB(tr1, tr4);
            output[3 * n_count] = VADD(tr1, tr4);

            input += 4 * count;
            output += count;
        }
        input = in;
        output = out;
    }
    if (count < 2) return;
    if (count != 2) {
        for (n = 0; n < n_count; n += count) {
            const v4sf *RESTRICT pc = (v4sf *) (input - 1 + 4 * n);
            v4sf *RESTRICT ph = output + n + 1;
            for (i = 2; i < count; i += 2) {
                tr1 = VSUB(pc[i], pc[4 * count - i]);
                tr2 = VADD(pc[i], pc[4 * count - i]);
                ti4 = VSUB(pc[2 * count + i], pc[2 * count - i]);
                tr3 = VADD(pc[2 * count + i], pc[2 * count - i]);
                ph[0] = VADD(tr2, tr3);
                cr3 = VSUB(tr2, tr3);

                ti3 = VSUB(pc[2 * count + i + 1], pc[2 * count - i + 1]);
                tr4 = VADD(pc[2 * count + i + 1], pc[2 * count - i + 1]);
                cr2 = VSUB(tr1, tr4);
                cr4 = VADD(tr1, tr4);

                ti1 = VADD(pc[i + 1], pc[4 * count - i + 1]);
                ti2 = VSUB(pc[i + 1], pc[4 * count - i + 1]);

                ph[1] = VADD(ti2, ti3);
                ph += n_count;
                ci3 = VSUB(ti2, ti3);
                ci2 = VADD(ti1, ti4);
                ci4 = VSUB(ti1, ti4);
                VCPLXMUL(cr2, ci2, LD_PS1(twiddle_1[i - 2]), LD_PS1(twiddle_1[i - 1]));
                ph[0] = cr2;
                ph[1] = ci2;
                ph += n_count;
                VCPLXMUL(cr3, ci3, LD_PS1(twiddle_2[i - 2]), LD_PS1(twiddle_2[i - 1]));
                ph[0] = cr3;
                ph[1] = ci3;
                ph += n_count;
                VCPLXMUL(cr4, ci4, LD_PS1(twiddle_3[i - 2]), LD_PS1(twiddle_3[i - 1]));
                ph[0] = cr4;
                ph[1] = ci4;
                ph = ph - 3 * n_count + 2;
            }
        }
        if (count % 2 == 1) return;
    }
    for (n = 0; n < n_count; n += count) {
        int j = 4 * n + count;
        v4sf c = input[j - 1], d = input[j + 2 * count - 1];
        v4sf a = input[j + 0], b = input[j + 2 * count + 0];
        tr1 = VSUB(c, d);
        tr2 = VADD(c, d);
        ti1 = VADD(b, a);
        ti2 = VSUB(b, a);
        output[count - 1 + n + 0 * n_count] = VADD(tr2, tr2);
        output[count - 1 + n + 1 * n_count] = SVMUL(minus_sqrt2, VSUB(ti1, tr1));
        output[count - 1 + n + 2 * n_count] = VADD(ti2, ti2);
        output[count - 1 + n + 3 * n_count] = SVMUL(minus_sqrt2, VADD(ti1, tr1));
    }
} /* radb4 */

static void dft_5_r(int count, int w_count, const v4sf *RESTRICT input, v4sf *RESTRICT output,
                    const float *twiddle_1, const float *twiddle_2, const float *twiddle_3, const float *twiddle_4) {
    static const float tr11 = .309016994374947f;
    static const float ti11 = .951056516295154f;
    static const float tr12 = -.809016994374947f;
    static const float ti12 = .587785252292473f;

    /* System generated locals */
    int in_offset, out_offset;

    /* Local variables */
    int i, n, j;
    v4sf ci2, di2, ci4, ci5, di3, di4, di5, ci3, cr2, cr3, dr2, dr3, dr4, dr5,
            cr5, cr4, ti2, ti3, ti5, ti4, tr2, tr3, tr4, tr5;
    int j2;

#define in_5r_ref(a_1, a_2, a_3) input[((a_3)*w_count + (a_2))*count + (a_1)]
#define out_5r_ref(a_1, a_2, a_3) output[((a_3)*5 + (a_2))*count + (a_1)]

    /* Parameter adjustments */
    out_offset = 1 + count * 6;
    output -= out_offset;
    in_offset = 1 + count * (1 + w_count);
    input -= in_offset;

    /* Function Body */
    for (n = 1; n <= w_count; ++n) {
        cr2 = VADD(in_5r_ref(1, n, 5), in_5r_ref(1, n, 2));
        ci5 = VSUB(in_5r_ref(1, n, 5), in_5r_ref(1, n, 2));
        cr3 = VADD(in_5r_ref(1, n, 4), in_5r_ref(1, n, 3));
        ci4 = VSUB(in_5r_ref(1, n, 4), in_5r_ref(1, n, 3));
        out_5r_ref(1, 1, n) = VADD(in_5r_ref(1, n, 1), VADD(cr2, cr3));
        out_5r_ref(count, 2, n) = VADD(in_5r_ref(1, n, 1), VADD(SVMUL(tr11, cr2), SVMUL(tr12, cr3)));
        out_5r_ref(1, 3, n) = VADD(SVMUL(ti11, ci5), SVMUL(ti12, ci4));
        out_5r_ref(count, 4, n) = VADD(in_5r_ref(1, n, 1), VADD(SVMUL(tr12, cr2), SVMUL(tr11, cr3)));
        out_5r_ref(1, 5, n) = VSUB(SVMUL(ti12, ci5), SVMUL(ti11, ci4));
        //printf("pffft: radf5, n=%d ch_ref=%f, ci4=%f\n", n, ch_ref(1, 5, n), ci4);
    }
    if (count == 1) {
        return;
    }
    j2 = count + 2;
    for (n = 1; n <= w_count; ++n) {
        for (i = 3; i <= count; i += 2) {
            j = j2 - i;
            dr2 = LD_PS1(twiddle_1[i - 3]);
            di2 = LD_PS1(twiddle_1[i - 2]);
            dr3 = LD_PS1(twiddle_2[i - 3]);
            di3 = LD_PS1(twiddle_2[i - 2]);
            dr4 = LD_PS1(twiddle_3[i - 3]);
            di4 = LD_PS1(twiddle_3[i - 2]);
            dr5 = LD_PS1(twiddle_4[i - 3]);
            di5 = LD_PS1(twiddle_4[i - 2]);
            VCPLXMULCONJ(dr2, di2, in_5r_ref(i - 1, n, 2), in_5r_ref(i, n, 2));
            VCPLXMULCONJ(dr3, di3, in_5r_ref(i - 1, n, 3), in_5r_ref(i, n, 3));
            VCPLXMULCONJ(dr4, di4, in_5r_ref(i - 1, n, 4), in_5r_ref(i, n, 4));
            VCPLXMULCONJ(dr5, di5, in_5r_ref(i - 1, n, 5), in_5r_ref(i, n, 5));
            cr2 = VADD(dr2, dr5);
            ci5 = VSUB(dr5, dr2);
            cr5 = VSUB(di2, di5);
            ci2 = VADD(di2, di5);
            cr3 = VADD(dr3, dr4);
            ci4 = VSUB(dr4, dr3);
            cr4 = VSUB(di3, di4);
            ci3 = VADD(di3, di4);
            out_5r_ref(i - 1, 1, n) = VADD(in_5r_ref(i - 1, n, 1), VADD(cr2, cr3));
            out_5r_ref(i, 1, n) = VSUB(in_5r_ref(i, n, 1), VADD(ci2, ci3));//
            tr2 = VADD(in_5r_ref(i - 1, n, 1), VADD(SVMUL(tr11, cr2), SVMUL(tr12, cr3)));
            ti2 = VSUB(in_5r_ref(i, n, 1), VADD(SVMUL(tr11, ci2), SVMUL(tr12, ci3)));//
            tr3 = VADD(in_5r_ref(i - 1, n, 1), VADD(SVMUL(tr12, cr2), SVMUL(tr11, cr3)));
            ti3 = VSUB(in_5r_ref(i, n, 1), VADD(SVMUL(tr12, ci2), SVMUL(tr11, ci3)));//
            tr5 = VADD(SVMUL(ti11, cr5), SVMUL(ti12, cr4));
            ti5 = VADD(SVMUL(ti11, ci5), SVMUL(ti12, ci4));
            tr4 = VSUB(SVMUL(ti12, cr5), SVMUL(ti11, cr4));
            ti4 = VSUB(SVMUL(ti12, ci5), SVMUL(ti11, ci4));
            out_5r_ref(i - 1, 3, n) = VSUB(tr2, tr5);
            out_5r_ref(j - 1, 2, n) = VADD(tr2, tr5);
            out_5r_ref(i, 3, n) = VADD(ti2, ti5);
            out_5r_ref(j, 2, n) = VSUB(ti5, ti2);
            out_5r_ref(i - 1, 5, n) = VSUB(tr3, tr4);
            out_5r_ref(j - 1, 4, n) = VADD(tr3, tr4);
            out_5r_ref(i, 5, n) = VADD(ti3, ti4);
            out_5r_ref(j, 4, n) = VSUB(ti4, ti3);
        }
    }
#undef in_5r_ref
#undef out_5r_ref
} /* radf5 */

static void idft_5_r(int count, int w_count, const v4sf *RESTRICT input, v4sf *RESTRICT output,
                     const float *twiddle_1, const float *twiddle_2, const float *twiddle_3, const float *twiddle_4) {
    static const float tr11 = .309016994374947f;
    static const float ti11 = .951056516295154f;
    static const float tr12 = -.809016994374947f;
    static const float ti12 = .587785252292473f;

    int in_offset, out_offset;

    /* Local variables */
    int i, n, j;
    v4sf ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, cr4, ti2, ti3,
            ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3, tr4, tr5;
    int j2;

#define in_i5r_ref(a_1, a_2, a_3) input[((a_3)*5 + (a_2))*count + (a_1)]
#define out_i5r_ref(a_1, a_2, a_3) output[((a_3)*w_count + (a_2))*count + (a_1)]

    /* Parameter adjustments */
    out_offset = 1 + count * (1 + w_count);
    output -= out_offset;
    in_offset = 1 + count * 6;
    input -= in_offset;

    /* Function Body */
    for (n = 1; n <= w_count; ++n) {
        ti5 = VADD(in_i5r_ref(1, 3, n), in_i5r_ref(1, 3, n));
        ti4 = VADD(in_i5r_ref(1, 5, n), in_i5r_ref(1, 5, n));
        tr2 = VADD(in_i5r_ref(count, 2, n), in_i5r_ref(count, 2, n));
        tr3 = VADD(in_i5r_ref(count, 4, n), in_i5r_ref(count, 4, n));
        out_i5r_ref(1, n, 1) = VADD(in_i5r_ref(1, 1, n), VADD(tr2, tr3));
        cr2 = VADD(in_i5r_ref(1, 1, n), VADD(SVMUL(tr11, tr2), SVMUL(tr12, tr3)));
        cr3 = VADD(in_i5r_ref(1, 1, n), VADD(SVMUL(tr12, tr2), SVMUL(tr11, tr3)));
        ci5 = VADD(SVMUL(ti11, ti5), SVMUL(ti12, ti4));
        ci4 = VSUB(SVMUL(ti12, ti5), SVMUL(ti11, ti4));
        out_i5r_ref(1, n, 2) = VSUB(cr2, ci5);
        out_i5r_ref(1, n, 3) = VSUB(cr3, ci4);
        out_i5r_ref(1, n, 4) = VADD(cr3, ci4);
        out_i5r_ref(1, n, 5) = VADD(cr2, ci5);
    }
    if (count == 1) {
        return;
    }
    j2 = count + 2;
    for (n = 1; n <= w_count; ++n) {
        for (i = 3; i <= count; i += 2) {
            j = j2 - i;
            ti5 = VADD(in_i5r_ref(i, 3, n), in_i5r_ref(j, 2, n));
            ti2 = VSUB(in_i5r_ref(i, 3, n), in_i5r_ref(j, 2, n));
            ti4 = VADD(in_i5r_ref(i, 5, n), in_i5r_ref(j, 4, n));
            ti3 = VSUB(in_i5r_ref(i, 5, n), in_i5r_ref(j, 4, n));
            tr5 = VSUB(in_i5r_ref(i - 1, 3, n), in_i5r_ref(j - 1, 2, n));
            tr2 = VADD(in_i5r_ref(i - 1, 3, n), in_i5r_ref(j - 1, 2, n));
            tr4 = VSUB(in_i5r_ref(i - 1, 5, n), in_i5r_ref(j - 1, 4, n));
            tr3 = VADD(in_i5r_ref(i - 1, 5, n), in_i5r_ref(j - 1, 4, n));
            out_i5r_ref(i - 1, n, 1) = VADD(in_i5r_ref(i - 1, 1, n), VADD(tr2, tr3));
            out_i5r_ref(i, n, 1) = VADD(in_i5r_ref(i, 1, n), VADD(ti2, ti3));
            cr2 = VADD(in_i5r_ref(i - 1, 1, n), VADD(SVMUL(tr11, tr2), SVMUL(tr12, tr3)));
            ci2 = VADD(in_i5r_ref(i, 1, n), VADD(SVMUL(tr11, ti2), SVMUL(tr12, ti3)));
            cr3 = VADD(in_i5r_ref(i - 1, 1, n), VADD(SVMUL(tr12, tr2), SVMUL(tr11, tr3)));
            ci3 = VADD(in_i5r_ref(i, 1, n), VADD(SVMUL(tr12, ti2), SVMUL(tr11, ti3)));
            cr5 = VADD(SVMUL(ti11, tr5), SVMUL(ti12, tr4));
            ci5 = VADD(SVMUL(ti11, ti5), SVMUL(ti12, ti4));
            cr4 = VSUB(SVMUL(ti12, tr5), SVMUL(ti11, tr4));
            ci4 = VSUB(SVMUL(ti12, ti5), SVMUL(ti11, ti4));
            dr3 = VSUB(cr3, ci4);
            dr4 = VADD(cr3, ci4);
            di3 = VADD(ci3, cr4);
            di4 = VSUB(ci3, cr4);
            dr5 = VADD(cr2, ci5);
            dr2 = VSUB(cr2, ci5);
            di5 = VSUB(ci2, cr5);
            di2 = VADD(ci2, cr5);
            VCPLXMUL(dr2, di2, LD_PS1(twiddle_1[i - 3]), LD_PS1(twiddle_1[i - 2]));
            VCPLXMUL(dr3, di3, LD_PS1(twiddle_2[i - 3]), LD_PS1(twiddle_2[i - 2]));
            VCPLXMUL(dr4, di4, LD_PS1(twiddle_3[i - 3]), LD_PS1(twiddle_3[i - 2]));
            VCPLXMUL(dr5, di5, LD_PS1(twiddle_4[i - 3]), LD_PS1(twiddle_4[i - 2]));

            out_i5r_ref(i - 1, n, 2) = dr2;
            out_i5r_ref(i, n, 2) = di2;
            out_i5r_ref(i - 1, n, 3) = dr3;
            out_i5r_ref(i, n, 3) = di3;
            out_i5r_ref(i - 1, n, 4) = dr4;
            out_i5r_ref(i, n, 4) = di4;
            out_i5r_ref(i - 1, n, 5) = dr5;
            out_i5r_ref(i, n, 5) = di5;
        }
    }
#undef in_i5r_ref
#undef out_i5r_ref
} /* radb5 */

static NEVER_INLINE(v4sf *) rfft_forward(size_t n, const v4sf *input_readonly, v4sf *work1, v4sf *work2,
                                         const float *twiddle, const int *radix) {
    v4sf *in = (v4sf *) input_readonly;
    v4sf *out = (in == work2 ? work1 : work2);
    int nr = radix[1], r;
    int offset = n;
    int w1 = n - 1;
    assert(in != out && work1 != work2);
    for (r = 1; r <= nr; ++r) {
        int i = nr - r;
        int rad = radix[i + 2];
        int w_count = offset / rad;
        int count = n / offset;
        w1 -= (rad - 1) * count;
        switch (rad) {
            case 5: {
                int w2 = w1 + count;
                int w3 = w2 + count;
                int w4 = w3 + count;
                dft_5_r(count, w_count, in, out, &twiddle[w1], &twiddle[w2], &twiddle[w3], &twiddle[w4]);
            }
                break;
            case 4: {
                int w2 = w1 + count;
                int w3 = w2 + count;
                dft_4_r(count, w_count, in, out, &twiddle[w1], &twiddle[w2], &twiddle[w3]);
            }
                break;
            case 3: {
                int w2 = w1 + count;
                dft_3_r(count, w_count, in, out, &twiddle[w1], &twiddle[w2]);
            }
                break;
            case 2:
                dft_2_r(count, w_count, in, out, &twiddle[w1]);
                break;
            default:
                assert(0);
                break;
        }
        offset = w_count;
        if (out == work2) {
            out = work1;
            in = work2;
        } else {
            out = work2;
            in = work1;
        }
    }
    return in; /* this is in fact the output .. */
} /* rfftf1 */

static NEVER_INLINE(v4sf *) rfft_backward(int n, const v4sf *input_readonly, v4sf *work1, v4sf *work2,
                                          const float *twiddle, const int *radix) {
    v4sf *in = (v4sf *) input_readonly;
    v4sf *out = (in == work2 ? work1 : work2);
    int nr = radix[1], r;
    int w_count = 1;
    int w1 = 0;
    assert(in != out);
    for (r = 1; r <= nr; r++) {
        int rad = radix[r + 1];
        int offset = rad * w_count;
        int count = n / offset;
        switch (rad) {
            case 5: {
                int w2 = w1 + count;
                int w3 = w2 + count;
                int w4 = w3 + count;
                idft_5_r(count, w_count, in, out, &twiddle[w1], &twiddle[w2], &twiddle[w3], &twiddle[w4]);
            }
                break;
            case 4: {
                int w2 = w1 + count;
                int w3 = w2 + count;
                idft_4_r(count, w_count, in, out, &twiddle[w1], &twiddle[w2], &twiddle[w3]);
            }
                break;
            case 3: {
                int w2 = w1 + count;
                idft_3_r(count, w_count, in, out, &twiddle[w1], &twiddle[w2]);
            }
                break;
            case 2:
                idft_2_r(count, w_count, in, out, &twiddle[w1]);
                break;
            default:
                assert(0);
                break;
        }
        w_count = offset;
        w1 += (rad - 1) * count;

        if (out == work2) {
            out = work1;
            in = work2;
        } else {
            out = work2;
            in = work1;
        }
    }
    return in; /* this is in fact the output .. */
}

static int decompose_radix(int n, int *radix, const int *stages) {
    int offset = n, stage = 0, i, s = 0;
    for (s = 0; stages[s]; ++s) {
        int rad = stages[s];
        while (offset != 1) {
            int w_count = offset / rad;
            int nr = offset - rad * w_count;
            if (nr == 0) {
                radix[2 + stage++] = rad;
                offset = w_count;
                if (rad == 2 && stage != 1) {
                    for (i = 2; i <= stage; ++i) {
                        int r = stage - i + 2;
                        radix[r + 1] = radix[r];
                    }
                    radix[2] = 2;
                }
            } else break;
        }
    }
    radix[0] = n;
    radix[1] = stage;
    return stage;
}

static void make_twiddles_r(int n, float *twiddles, int *radix) {
    static const int stages[] = {4, 2, 3, 5, 0};
    int s, r, i;

    int stage = decompose_radix(n, radix, stages);
    double w_pi = (2 * M_PI) / n;
    int n_count = 0;
    int cur_stage = stage - 1;
    int w_count = 1;
    for (s = 1; s <= cur_stage; s++) {
        int rad = radix[s + 1];
        int w = 0;
        int offset = w_count * rad;
        int count = n / offset;
        int cur_rad = rad - 1;
        for (r = 1; r <= cur_rad; ++r) {
            int ni = n_count, fi = 0;
            w += w_count;
            double wexp = w * w_pi;
            for (i = 3; i <= count; i += 2) {
                ni += 2;
                fi += 1;
                SinCos(fi * wexp, &twiddles[ni - 1], &twiddles[ni - 2]);
            }
            n_count += count;
        }
        w_count = offset;
    }
} /* rffti1 */

void make_twiddles_c(int n, float *twiddle, int *radix) {
    static const int stages[] = {5, 3, 4, 2, 0};
    int s, r, i;

    int stage = decompose_radix(n, radix, stages);
    double w_pi = (2 * M_PI) / n;
    int ni = 1;
    int w_count = 1;
    for (s = 1; s <= stage; s++) {
        int rad = radix[s + 1];
        int w = 0;
        int offset = w_count * rad;
        int count = n / offset;
        int n_count = count + count + 2;
        int cur_rad = rad - 1;
        for (r = 1; r <= cur_rad; r++) {
            int j = ni, fi = 0;
            twiddle[ni - 1] = 1;
            twiddle[ni] = 0;
            w += w_count;
            double wexp = w * w_pi;
            for (i = 4; i <= n_count; i += 2) {
                ni += 2;
                fi += 1;
                SinCos(fi * wexp, &twiddle[ni], &twiddle[ni - 1]);
            }
            if (rad > 5) {
                twiddle[j - 1] = twiddle[ni - 1];
                twiddle[j] = twiddle[ni];
            }
        }
        w_count = offset;
    }
} /* cffti1 */

v4sf *
cfft(size_t n, const v4sf *input_readonly, v4sf *work1, v4sf *work2, const float *twiddle, const int *radix,
     int isign) {
    v4sf *in = (v4sf *) input_readonly;
    v4sf *out = (in == work2 ? work1 : work2);
    int nr = radix[1], r;
    int w_count = 1;
    int w1 = 0;
    assert(in != out && work1 != work2);
    for (r = 2; r <= nr + 1; r++) {
        int rad = radix[r];
        int offset = rad * w_count;
        int half_count = n / offset;
        int count = half_count + half_count;
        switch (rad) {
            case 5: {
                int w2 = w1 + count;
                int w3 = w2 + count;
                int w4 = w3 + count;
                cdft_5(count, w_count, in, out, &twiddle[w1], &twiddle[w2], &twiddle[w3], &twiddle[w4], isign);
            }
                break;
            case 4: {
                int w2 = w1 + count;
                int w3 = w2 + count;
                cdft_4(count, w_count, in, out, &twiddle[w1], &twiddle[w2], &twiddle[w3], isign);
            }
                break;
            case 2: {
                cdft_2(count, w_count, in, out, &twiddle[w1], isign);
            }
                break;
            case 3: {
                int w2 = w1 + count;
                cdft_3(count, w_count, in, out, &twiddle[w1], &twiddle[w2], isign);
            }
                break;
            default:
                assert(0);
        }
        w_count = offset;
        w1 += (rad - 1) * count;
        if (out == work2) {
            out = work1;
            in = work2;
        } else {
            out = work2;
            in = work1;
        }
    }

    return in; /* this is in fact the output .. */
}

struct PFFFT_Setup {
    size_t N;
    size_t vec_size; // nb of complex simd vectors (N/4 if PFFFT_COMPLEX, N/8 if PFFFT_REAL)
    int radix[15];
    pffft_transform_t transform;
    v4sf *data; // allocated room for twiddle coefs
    float *workset;    // points into 'data' , N/4*3 elements
    float *twiddle; // points into 'data', N/4 elements
};

int is_valid_size(size_t n) {
    while (n > 1) {
        // premade codelets 2-5
        unsigned factor = 5;
        for (; factor > 1; factor--) {
            if (!(n % factor)) {
                break;
            }
        }
        if (factor == 1) {
            return -1;
        }
        n /= factor;
    }
    return 1;
}

int is_valid_param(size_t n, pffft_transform_t transform) {
    if (is_valid_size(n) == -1)
        return -1;
    /* unfortunately, the fft size must be a multiple of 16 for complex FFTs
       and 32 for real FFTs -- a lot of stuff would need to be rewritten to
       handle other cases (or maybe just switch to a scalar fft, I don't know..) */
    if (transform == PFFFT_REAL) {
        if (!((n % (2 * SIMD_SZ * SIMD_SZ)) == 0 && n > 0))
            return -1;
    }
    if (transform == PFFFT_COMPLEX) {
        if (!((n % (SIMD_SZ * SIMD_SZ)) == 0 && n > 0))
            return -1;
    }
    return 1;
}

PFFFT_Setup *pffft_new_setup(int N, pffft_transform_t transform) {
    if (is_valid_param(N, transform) != 1)
        return NULL;
    /* unfortunately, the fft size must be a multiple of 16 for complex FFTs
        and 32 for real FFTs -- a lot of stuff would need to be rewritten to
        handle other cases (or maybe just switch to a scalar fft, I don't know..) */
    // if (transform == PFFFT_REAL) { assert((N % (2 * SIMD_SZ * SIMD_SZ)) == 0 && N > 0); } if
    // (transform == PFFFT_COMPLEX) { assert((N % (SIMD_SZ * SIMD_SZ)) == 0 && N > 0); }

    //assert((N % 32) == 0);
    PFFFT_Setup *s = (PFFFT_Setup *) pffft_aligned_calloc(1, sizeof(PFFFT_Setup));
    if (s == NULL)
        return NULL;
    int k, m;
    s->N = N;
    s->transform = transform;
    /* nb of complex simd vectors */
    s->vec_size = (transform == PFFFT_REAL ? N / 2 : N) / SIMD_SZ;
    s->data = (v4sf *) pffft_aligned_calloc(2 * s->vec_size, sizeof(v4sf));
    if (s->data == NULL) {
        pffft_aligned_free(s);
        return NULL;
    }
    s->workset = (float *) s->data;
    s->twiddle = (float *) (s->data + (2 * s->vec_size * (SIMD_SZ - 1)) / SIMD_SZ);

    if (transform == PFFFT_REAL) {
        for (k = 0; k < s->vec_size; ++k) {
            int i = k / SIMD_SZ;
            int j = k % SIMD_SZ;
            for (m = 0; m < SIMD_SZ - 1; ++m) {
                SinCos(-2 * M_PI * (m + 1) * k / N, &s->workset[(2 * (i * 3 + m) + 1) * SIMD_SZ + j],
                       &s->workset[(2 * (i * 3 + m) + 0) * SIMD_SZ + j]);
            }
        }
        make_twiddles_r(N / SIMD_SZ, s->twiddle, s->radix);
    } else {
        for (k = 0; k < s->vec_size; ++k) {
            int i = k / SIMD_SZ;
            int j = k % SIMD_SZ;
            for (m = 0; m < SIMD_SZ - 1; ++m) {
                SinCos(-2 * M_PI * (m + 1) * k / N, &s->workset[(2 * (i * 3 + m) + 1) * SIMD_SZ + j],
                       &s->workset[(2 * (i * 3 + m) + 0) * SIMD_SZ + j]);
            }
        }
        make_twiddles_c(N / SIMD_SZ, s->twiddle, s->radix);
    }

    /* check that N is decomposable with allowed prime factors */
    for (k = 0, m = 1; k < s->radix[1]; ++k) {
        m *= s->radix[2 + k];
    }
    if (m != N / SIMD_SZ) {
        pffft_destroy_setup(s);
        s = 0;
    }

    return s;
}

void pffft_destroy_setup(PFFFT_Setup *s) {
    if (s == NULL)
        return;
    pffft_aligned_free(s->data);
    pffft_aligned_free(s);
}

#if !defined(PFFFT_SIMD_DISABLE)

/* [0 0 1 2 3 4 5 6 7 8] -> [0 8 7 6 5 4 3 2 1] */
static void reversed_copy(int N, const v4sf *in, int in_stride, v4sf *out) {
    v4sf g0, g1;
    int k;
    INTERLEAVE2(in[0], in[1], g0, g1);
    in += in_stride;

    *--out = VSWAPHL(g0, g1); // [g0l, g0h], [g1l g1h] -> [g1l, g0h]
    for (k = 1; k < N; ++k) {
        v4sf h0, h1;
        INTERLEAVE2(in[0], in[1], h0, h1);
        in += in_stride;
        *--out = VSWAPHL(g1, h0);
        *--out = VSWAPHL(h0, h1);
        g1 = h1;
    }
    *--out = VSWAPHL(g1, g0);
}

static void unreversed_copy(int N, const v4sf *in, v4sf *out, int out_stride) {
    v4sf g0, g1, h0, h1;
    int k;
    g0 = g1 = in[0];
    ++in;
    for (k = 1; k < N; ++k) {
        h0 = *in++;
        h1 = *in++;
        g1 = VSWAPHL(g1, h0);
        h0 = VSWAPHL(h0, h1);
        UNINTERLEAVE2(h0, g1, out[0], out[1]);
        out += out_stride;
        g1 = h1;
    }
    h0 = *in++;
    h1 = g0;
    g1 = VSWAPHL(g1, h0);
    h0 = VSWAPHL(h0, h1);
    UNINTERLEAVE2(h0, g1, out[0], out[1]);
}

void pffft_zreorder(PFFFT_Setup *setup, const float *in, float *out, pffft_direction_t direction) {
    int N = setup->N, vec_size = setup->vec_size;
    const v4sf *vin = (v4sf *) in;
    v4sf *vout = (v4sf *) out;
    assert(in != out);
    if (setup->transform == PFFFT_REAL) {
        int k, dk = N / 32;
        if (direction == PFFFT_FORWARD) {
            for (k = 0; k < dk; ++k) {
                INTERLEAVE2(vin[k * 8 + 0], vin[k * 8 + 1], vout[2 * (0 * dk + k) + 0], vout[2 * (0 * dk + k) + 1]);
                INTERLEAVE2(vin[k * 8 + 4], vin[k * 8 + 5], vout[2 * (2 * dk + k) + 0], vout[2 * (2 * dk + k) + 1]);
            }
            reversed_copy(dk, vin + 2, 8, (v4sf *) out + N / 2);
            reversed_copy(dk, vin + 6, 8, (v4sf *) out + N);
        } else {
            for (k = 0; k < dk; ++k) {
                UNINTERLEAVE2(vin[2 * (0 * dk + k) + 0], vin[2 * (0 * dk + k) + 1], vout[k * 8 + 0], vout[k * 8 + 1]);
                UNINTERLEAVE2(vin[2 * (2 * dk + k) + 0], vin[2 * (2 * dk + k) + 1], vout[k * 8 + 4], vout[k * 8 + 5]);
            }
            unreversed_copy(dk, (v4sf *) (in + N / 4), (v4sf *) out + N - 6 * SIMD_SZ, -8);
            unreversed_copy(dk, (v4sf *) (in + 3 * N / 4), (v4sf *) out + N - 2 * SIMD_SZ, -8);
        }
    } else {
        int k;
        if (direction == PFFFT_FORWARD) {
            for (k = 0; k < vec_size; ++k) {
                int kk = (k / 4) + (k % 4) * (vec_size / 4);
                INTERLEAVE2(vin[k * 2], vin[k * 2 + 1], vout[kk * 2], vout[kk * 2 + 1]);
            }
        } else {
            for (k = 0; k < vec_size; ++k) {
                int kk = (k / 4) + (k % 4) * (vec_size / 4);
                UNINTERLEAVE2(vin[kk * 2], vin[kk * 2 + 1], vout[k * 2], vout[k * 2 + 1]);
            }
        }
    }
}

void pffft_cplx_finalize(int vec_size, const v4sf *in, v4sf *out, const v4sf *e) {
    int k, dk = vec_size / SIMD_SZ; // number of 4x4 matrix blocks
    v4sf r0, i0, r1, i1, r2, i2, r3, i3;
    v4sf sr0, dr0, sr1, dr1, si0, di0, si1, di1;
    assert(in != out);
    for (k = 0; k < dk; ++k) {
        r0 = in[8 * k + 0];
        i0 = in[8 * k + 1];
        r1 = in[8 * k + 2];
        i1 = in[8 * k + 3];
        r2 = in[8 * k + 4];
        i2 = in[8 * k + 5];
        r3 = in[8 * k + 6];
        i3 = in[8 * k + 7];
        VTRANSPOSE4(r0, r1, r2, r3);
        VTRANSPOSE4(i0, i1, i2, i3);
        VCPLXMUL(r1, i1, e[k * 6 + 0], e[k * 6 + 1]);
        VCPLXMUL(r2, i2, e[k * 6 + 2], e[k * 6 + 3]);
        VCPLXMUL(r3, i3, e[k * 6 + 4], e[k * 6 + 5]);

        sr0 = VADD(r0, r2);
        dr0 = VSUB(r0, r2);
        sr1 = VADD(r1, r3);
        dr1 = VSUB(r1, r3);
        si0 = VADD(i0, i2);
        di0 = VSUB(i0, i2);
        si1 = VADD(i1, i3);
        di1 = VSUB(i1, i3);

        /*
          transformation for each column is:

          [1   1   1   1   0   0   0   0]   [r0]
          [1   0  -1   0   0  -1   0   1]   [r1]
          [1  -1   1  -1   0   0   0   0]   [r2]
          [1   0  -1   0   0   1   0  -1]   [r3]
          [0   0   0   0   1   1   1   1] * [i0]
          [0   1   0  -1   1   0  -1   0]   [i1]
          [0   0   0   0   1  -1   1  -1]   [i2]
          [0  -1   0   1   1   0  -1   0]   [i3]
        */

        r0 = VADD(sr0, sr1);
        i0 = VADD(si0, si1);
        r1 = VADD(dr0, di1);
        i1 = VSUB(di0, dr1);
        r2 = VSUB(sr0, sr1);
        i2 = VSUB(si0, si1);
        r3 = VSUB(dr0, di1);
        i3 = VADD(di0, dr1);

        *out++ = r0;
        *out++ = i0;
        *out++ = r1;
        *out++ = i1;
        *out++ = r2;
        *out++ = i2;
        *out++ = r3;
        *out++ = i3;
    }
}

void pffft_cplx_preprocess(int vec_size, const v4sf *in, v4sf *out, const v4sf *workset) {
    int k, dk = vec_size / SIMD_SZ; // number of 4x4 matrix blocks
    v4sf r0, i0, r1, i1, r2, i2, r3, i3;
    v4sf sr0, dr0, sr1, dr1, si0, di0, si1, di1;
    assert(in != out);
    for (k = 0; k < dk; ++k) {
        r0 = in[8 * k + 0];
        i0 = in[8 * k + 1];
        r1 = in[8 * k + 2];
        i1 = in[8 * k + 3];
        r2 = in[8 * k + 4];
        i2 = in[8 * k + 5];
        r3 = in[8 * k + 6];
        i3 = in[8 * k + 7];

        sr0 = VADD(r0, r2);
        dr0 = VSUB(r0, r2);
        sr1 = VADD(r1, r3);
        dr1 = VSUB(r1, r3);
        si0 = VADD(i0, i2);
        di0 = VSUB(i0, i2);
        si1 = VADD(i1, i3);
        di1 = VSUB(i1, i3);

        r0 = VADD(sr0, sr1);
        i0 = VADD(si0, si1);
        r1 = VSUB(dr0, di1);
        i1 = VADD(di0, dr1);
        r2 = VSUB(sr0, sr1);
        i2 = VSUB(si0, si1);
        r3 = VADD(dr0, di1);
        i3 = VSUB(di0, dr1);

        VCPLXMULCONJ(r1, i1, workset[k * 6 + 0], workset[k * 6 + 1]);
        VCPLXMULCONJ(r2, i2, workset[k * 6 + 2], workset[k * 6 + 3]);
        VCPLXMULCONJ(r3, i3, workset[k * 6 + 4], workset[k * 6 + 5]);

        VTRANSPOSE4(r0, r1, r2, r3);
        VTRANSPOSE4(i0, i1, i2, i3);

        *out++ = r0;
        *out++ = i0;
        *out++ = r1;
        *out++ = i1;
        *out++ = r2;
        *out++ = i2;
        *out++ = r3;
        *out++ = i3;
    }
}

static ALWAYS_INLINE(void) pffft_real_finalize_4x4(const v4sf *in0, const v4sf *in1, const v4sf *in,
                                                   const v4sf *workset, v4sf *out) {
    v4sf r0, i0, r1, i1, r2, i2, r3, i3;
    v4sf sr0, dr0, sr1, dr1, si0, di0, si1, di1;
    r0 = *in0;
    i0 = *in1;
    r1 = *in++;
    i1 = *in++;
    r2 = *in++;
    i2 = *in++;
    r3 = *in++;
    i3 = *in++;
    VTRANSPOSE4(r0, r1, r2, r3);
    VTRANSPOSE4(i0, i1, i2, i3);

    /*
      transformation for each column is:

      [1   1   1   1   0   0   0   0]   [r0]
      [1   0  -1   0   0  -1   0   1]   [r1]
      [1   0  -1   0   0   1   0  -1]   [r2]
      [1  -1   1  -1   0   0   0   0]   [r3]
      [0   0   0   0   1   1   1   1] * [i0]
      [0  -1   0   1  -1   0   1   0]   [i1]
      [0  -1   0   1   1   0  -1   0]   [i2]
      [0   0   0   0  -1   1  -1   1]   [i3]
    */

    //cerr << "matrix initial, before workset , REAL:\n 1: " << r0 << "\n 1: " << r1 << "\n 1: " << r2 << "\n 1: " << r3 << "\n";
    //cerr << "matrix initial, before workset, IMAG :\n 1: " << i0 << "\n 1: " << i1 << "\n 1: " << i2 << "\n 1: " << i3 << "\n";

    VCPLXMUL(r1, i1, workset[0], workset[1]);
    VCPLXMUL(r2, i2, workset[2], workset[3]);
    VCPLXMUL(r3, i3, workset[4], workset[5]);

    //cerr << "matrix initial, real part:\n 1: " << r0 << "\n 1: " << r1 << "\n 1: " << r2 << "\n 1: " << r3 << "\n";
    //cerr << "matrix initial, imag part:\n 1: " << i0 << "\n 1: " << i1 << "\n 1: " << i2 << "\n 1: " << i3 << "\n";

    sr0 = VADD(r0, r2);
    dr0 = VSUB(r0, r2);
    sr1 = VADD(r1, r3);
    dr1 = VSUB(r3, r1);
    si0 = VADD(i0, i2);
    di0 = VSUB(i0, i2);
    si1 = VADD(i1, i3);
    di1 = VSUB(i3, i1);

    r0 = VADD(sr0, sr1);
    r3 = VSUB(sr0, sr1);
    i0 = VADD(si0, si1);
    i3 = VSUB(si1, si0);
    r1 = VADD(dr0, di1);
    r2 = VSUB(dr0, di1);
    i1 = VSUB(dr1, di0);
    i2 = VADD(dr1, di0);

    *out++ = r0;
    *out++ = i0;
    *out++ = r1;
    *out++ = i1;
    *out++ = r2;
    *out++ = i2;
    *out++ = r3;
    *out++ = i3;
}

static NEVER_INLINE(void) pffft_real_finalize(int vec_size, const v4sf *in, v4sf *out, const v4sf *workset) {
    int k, dk = vec_size / SIMD_SZ; // number of 4x4 matrix blocks
    /* fftpack order is f0r f1r f1i f2r f2i ... f(n-1)r f(n-1)i f(n)r */

    v4sf_union cr, ci, *uout = (v4sf_union *) out;
    v4sf save = in[7], zero = VZERO();
    float xr0, xi0, xr1, xi1, xr2, xi2, xr3, xi3;
    static const float s = (M_SQRT2 / 2);

    cr.v = in[0];
    ci.v = in[vec_size * 2 - 1];
    assert(in != out);
    pffft_real_finalize_4x4(&zero, &zero, in + 1, workset, out);

    /*
      [cr0 cr1 cr2 cr3 ci0 ci1 ci2 ci3]

      [Xr(1)]  ] [1   1   1   1   0   0   0   0]
      [Xr(N/4) ] [0   0   0   0   1   s   0  -s]
      [Xr(N/2) ] [1   0  -1   0   0   0   0   0]
      [Xr(3N/4)] [0   0   0   0   1  -s   0   s]
      [Xi(1)   ] [1  -1   1  -1   0   0   0   0]
      [Xi(N/4) ] [0   0   0   0   0  -s  -1  -s]
      [Xi(N/2) ] [0  -1   0   1   0   0   0   0]
      [Xi(3N/4)] [0   0   0   0   0  -s   1  -s]
    */

    xr0 = (cr.f[0] + cr.f[2]) + (cr.f[1] + cr.f[3]);
    uout[0].f[0] = xr0;
    xi0 = (cr.f[0] + cr.f[2]) - (cr.f[1] + cr.f[3]);
    uout[1].f[0] = xi0;
    xr2 = (cr.f[0] - cr.f[2]);
    uout[4].f[0] = xr2;
    xi2 = (cr.f[3] - cr.f[1]);
    uout[5].f[0] = xi2;
    xr1 = ci.f[0] + s * (ci.f[1] - ci.f[3]);
    uout[2].f[0] = xr1;
    xi1 = -ci.f[2] - s * (ci.f[1] + ci.f[3]);
    uout[3].f[0] = xi1;
    xr3 = ci.f[0] - s * (ci.f[1] - ci.f[3]);
    uout[6].f[0] = xr3;
    xi3 = ci.f[2] - s * (ci.f[1] + ci.f[3]);
    uout[7].f[0] = xi3;

    for (k = 1; k < dk; ++k) {
        v4sf save_next = in[8 * k + 7];
        pffft_real_finalize_4x4(&save, &in[8 * k + 0], in + 8 * k + 1,
                                workset + k * 6, out + k * 8);
        save = save_next;
    }
}

static ALWAYS_INLINE(void) pffft_real_preprocess_4x4(const v4sf *in,
                                                     const v4sf *workset, v4sf *out, int first) {
    v4sf r0 = in[0], i0 = in[1], r1 = in[2], i1 = in[3], r2 = in[4], i2 = in[5], r3 = in[6], i3 = in[7];
    /*
      transformation for each column is:

      [1   1   1   1   0   0   0   0]   [r0]
      [1   0   0  -1   0  -1  -1   0]   [r1]
      [1  -1  -1   1   0   0   0   0]   [r2]
      [1   0   0  -1   0   1   1   0]   [r3]
      [0   0   0   0   1  -1   1  -1] * [i0]
      [0  -1   1   0   1   0   0   1]   [i1]
      [0   0   0   0   1   1  -1  -1]   [i2]
      [0   1  -1   0   1   0   0   1]   [i3]
    */

    v4sf sr0 = VADD(r0, r3), dr0 = VSUB(r0, r3);
    v4sf sr1 = VADD(r1, r2), dr1 = VSUB(r1, r2);
    v4sf si0 = VADD(i0, i3), di0 = VSUB(i0, i3);
    v4sf si1 = VADD(i1, i2), di1 = VSUB(i1, i2);

    r0 = VADD(sr0, sr1);
    r2 = VSUB(sr0, sr1);
    r1 = VSUB(dr0, si1);
    r3 = VADD(dr0, si1);
    i0 = VSUB(di0, di1);
    i2 = VADD(di0, di1);
    i1 = VSUB(si0, dr1);
    i3 = VADD(si0, dr1);

    VCPLXMULCONJ(r1, i1, workset[0], workset[1]);
    VCPLXMULCONJ(r2, i2, workset[2], workset[3]);
    VCPLXMULCONJ(r3, i3, workset[4], workset[5]);

    VTRANSPOSE4(r0, r1, r2, r3);
    VTRANSPOSE4(i0, i1, i2, i3);

    if (!first) {
        *out++ = r0;
        *out++ = i0;
    }
    *out++ = r1;
    *out++ = i1;
    *out++ = r2;
    *out++ = i2;
    *out++ = r3;
    *out++ = i3;
}

static NEVER_INLINE(void) pffft_real_preprocess(int vec_size, const v4sf *in, v4sf *out, const v4sf *workset) {
    int k, dk = vec_size / SIMD_SZ; // number of 4x4 matrix blocks
    /* fftpack order is f0r f1r f1i f2r f2i ... f(n-1)r f(n-1)i f(n)r */

    v4sf_union Xr, Xi, *uout = (v4sf_union *) out;
    float cr0, ci0, cr1, ci1, cr2, ci2, cr3, ci3;
    static const float s = M_SQRT2;
    assert(in != out);
    for (k = 0; k < 4; ++k) {
        Xr.f[k] = ((float *) in)[8 * k];
        Xi.f[k] = ((float *) in)[8 * k + 4];
    }

    pffft_real_preprocess_4x4(in, workset, out + 1, 1); // will write only 6 values

    /*
      [Xr0 Xr1 Xr2 Xr3 Xi0 Xi1 Xi2 Xi3]

      [cr0] [1   0   2   0   1   0   0   0]
      [cr1] [1   0   0   0  -1   0  -2   0]
      [cr2] [1   0  -2   0   1   0   0   0]
      [cr3] [1   0   0   0  -1   0   2   0]
      [ci0] [0   2   0   2   0   0   0   0]
      [ci1] [0   s   0  -s   0  -s   0  -s]
      [ci2] [0   0   0   0   0  -2   0   2]
      [ci3] [0  -s   0   s   0  -s   0  -s]
    */
    for (k = 1; k < dk; ++k) {
        pffft_real_preprocess_4x4(in + 8 * k, workset + k * 6, out - 1 + k * 8, 0);
    }

    cr0 = (Xr.f[0] + Xi.f[0]) + 2 * Xr.f[2];
    uout[0].f[0] = cr0;
    cr1 = (Xr.f[0] - Xi.f[0]) - 2 * Xi.f[2];
    uout[0].f[1] = cr1;
    cr2 = (Xr.f[0] + Xi.f[0]) - 2 * Xr.f[2];
    uout[0].f[2] = cr2;
    cr3 = (Xr.f[0] - Xi.f[0]) + 2 * Xi.f[2];
    uout[0].f[3] = cr3;
    ci0 = 2 * (Xr.f[1] + Xr.f[3]);
    uout[2 * vec_size - 1].f[0] = ci0;
    ci1 = s * (Xr.f[1] - Xr.f[3]) - s * (Xi.f[1] + Xi.f[3]);
    uout[2 * vec_size - 1].f[1] = ci1;
    ci2 = 2 * (Xi.f[3] - Xi.f[1]);
    uout[2 * vec_size - 1].f[2] = ci2;
    ci3 = -s * (Xr.f[1] - Xr.f[3]) - s * (Xi.f[1] + Xi.f[3]);
    uout[2 * vec_size - 1].f[3] = ci3;
}

void pffft_transform_internal(PFFFT_Setup *setup, const float *finput, float *foutput, v4sf *scratch,
                              pffft_direction_t direction, int ordered) {
    int k, vec_size = setup->vec_size;
    int nf_odd = (setup->radix[1] & 1);

    // temporary buffer is allocated on the stack if the scratch pointer is NULL
    int stack_allocate = (scratch == 0 ? vec_size * 2 : 1);
    VLA_ARRAY_ON_STACK(v4sf, scratch_on_stack, stack_allocate);

    const v4sf *vinput = (v4sf *) finput;
    v4sf *voutput = (v4sf *) foutput;
    v4sf *buff[2] = {voutput, scratch ? scratch : scratch_on_stack};
    int ib = (nf_odd ^ ordered ? 1 : 0);

    assert(VALIGNED(finput) && VALIGNED(foutput));

    //assert(finput != foutput);
    if (direction == PFFFT_FORWARD) {
        ib = !ib;
        if (setup->transform == PFFFT_REAL) {
            ib = (rfft_forward(vec_size * 2, vinput, buff[ib], buff[!ib],
                               setup->twiddle, &setup->radix[0]) == buff[0] ? 0 : 1);
            pffft_real_finalize(vec_size, buff[ib], buff[!ib], (v4sf *) setup->workset);
        } else {
            v4sf *tmp = buff[ib];
            for (k = 0; k < vec_size; ++k) {
                UNINTERLEAVE2(vinput[k * 2], vinput[k * 2 + 1], tmp[k * 2], tmp[k * 2 + 1]);
            }
            ib = (cfft(vec_size, buff[ib], buff[!ib], buff[ib],
                       setup->twiddle, &setup->radix[0], -1) == buff[0] ? 0 : 1);
            pffft_cplx_finalize(vec_size, buff[ib], buff[!ib], (v4sf *) setup->workset);
        }
        if (ordered) {
            pffft_zreorder(setup, (float *) buff[!ib], (float *) buff[ib], PFFFT_FORWARD);
        } else ib = !ib;
    } else {
        if (vinput == buff[ib]) {
            ib = !ib; // may happen when finput == foutput
        }
        if (ordered) {
            pffft_zreorder(setup, (float *) vinput, (float *) buff[ib], PFFFT_BACKWARD);
            vinput = buff[ib];
            ib = !ib;
        }
        if (setup->transform == PFFFT_REAL) {
            pffft_real_preprocess(vec_size, vinput, buff[ib], (v4sf *) setup->workset);
            ib = (rfft_backward(vec_size * 2, buff[ib], buff[0], buff[1],
                                setup->twiddle, &setup->radix[0]) == buff[0] ? 0 : 1);
        } else {
            pffft_cplx_preprocess(vec_size, vinput, buff[ib], (v4sf *) setup->workset);
            ib = (cfft(vec_size, buff[ib], buff[0], buff[1],
                       setup->twiddle, &setup->radix[0], +1) == buff[0] ? 0 : 1);
            for (k = 0; k < vec_size; ++k) {
                INTERLEAVE2(buff[ib][k * 2], buff[ib][k * 2 + 1], buff[ib][k * 2], buff[ib][k * 2 + 1]);
            }
        }
    }

    if (buff[ib] != voutput) {
        /* extra copy required -- this situation should only happen when finput == foutput */
        assert(finput == foutput);
        for (k = 0; k < vec_size; ++k) {
            v4sf a = buff[ib][2 * k], b = buff[ib][2 * k + 1];
            voutput[2 * k] = a;
            voutput[2 * k + 1] = b;
        }
        ib = !ib;
    }
    assert(buff[ib] == voutput);
}

void pffft_zconvolve_accumulate(PFFFT_Setup *s, const float *a, const float *b, float *ab, float scaling) {
    int vec_size = s->vec_size;
    const v4sf *RESTRICT va = (v4sf *) a;
    const v4sf *RESTRICT vb = (v4sf *) b;
    v4sf *RESTRICT vab = (v4sf *) ab;

#ifdef __arm__
    __builtin_prefetch(va);
    __builtin_prefetch(vb);
    __builtin_prefetch(vab);
    __builtin_prefetch(va + 2);
    __builtin_prefetch(vb + 2);
    __builtin_prefetch(vab + 2);
    __builtin_prefetch(va + 4);
    __builtin_prefetch(vb + 4);
    __builtin_prefetch(vab + 4);
    __builtin_prefetch(va + 6);
    __builtin_prefetch(vb + 6);
    __builtin_prefetch(vab + 6);
# ifndef __clang__
#   define ZCONVOLVE_USING_INLINE_NEON_ASM
# endif
#endif

    float ar, ai, br, bi, abr, abi;
#ifndef ZCONVOLVE_USING_INLINE_ASM
    v4sf vscal = LD_PS1(scaling);
    int i;
#endif

    assert(VALIGNED(a) && VALIGNED(b) && VALIGNED(ab));
    ar = ((v4sf_union *) va)[0].f[0];
    ai = ((v4sf_union *) va)[1].f[0];
    br = ((v4sf_union *) vb)[0].f[0];
    bi = ((v4sf_union *) vb)[1].f[0];
    abr = ((v4sf_union *) vab)[0].f[0];
    abi = ((v4sf_union *) vab)[1].f[0];

#ifdef ZCONVOLVE_USING_INLINE_ASM // inline asm version, unfortunately miscompiled by clang 3.2, at least on ubuntu.. so this will be restricted to gcc
    const float *a_ = a, *b_ = b; float *ab_ = ab;
    int N = vec_size;
    asm volatile("mov         r8, %2                  \n"
        "vdup.f32    q15, %4                 \n"
        "1:                                  \n"
        "pld         [%0,#64]                \n"
        "pld         [%1,#64]                \n"
        "pld         [%2,#64]                \n"
        "pld         [%0,#96]                \n"
        "pld         [%1,#96]                \n"
        "pld         [%2,#96]                \n"
        "vld1.f32    {q0,q1},   [%0,:128]!         \n"
        "vld1.f32    {q4,q5},   [%1,:128]!         \n"
        "vld1.f32    {q2,q3},   [%0,:128]!         \n"
        "vld1.f32    {q6,q7},   [%1,:128]!         \n"
        "vld1.f32    {q8,q9},   [r8,:128]!          \n"

        "vmul.f32    q10, q0, q4             \n"
        "vmul.f32    q11, q0, q5             \n"
        "vmul.f32    q12, q2, q6             \n"
        "vmul.f32    q13, q2, q7             \n"
        "vmls.f32    q10, q1, q5             \n"
        "vmla.f32    q11, q1, q4             \n"
        "vld1.f32    {q0,q1}, [r8,:128]!     \n"
        "vmls.f32    q12, q3, q7             \n"
        "vmla.f32    q13, q3, q6             \n"
        "vmla.f32    q8, q10, q15            \n"
        "vmla.f32    q9, q11, q15            \n"
        "vmla.f32    q0, q12, q15            \n"
        "vmla.f32    q1, q13, q15            \n"
        "vst1.f32    {q8,q9},[%2,:128]!    \n"
        "vst1.f32    {q0,q1},[%2,:128]!    \n"
        "subs        %3, #2                  \n"
        "bne         1b                      \n"
        : "+r"(a_), "+r"(b_), "+r"(ab_), "+r"(N) : "r"(scaling) : "r8", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q15", "memory");
#else // default routine, works fine for non-arm cpus with current compilers
    for (i = 0; i < vec_size; i += 2) {
        v4sf _ar, _ai, _br, _bi;
        _ar = va[2 * i + 0];
        _ai = va[2 * i + 1];
        _br = vb[2 * i + 0];
        _bi = vb[2 * i + 1];
        VCPLXMUL(_ar, _ai, _br, _bi);
        vab[2 * i + 0] = VMADD(_ar, vscal, vab[2 * i + 0]);
        vab[2 * i + 1] = VMADD(_ai, vscal, vab[2 * i + 1]);
        _ar = va[2 * i + 2];
        _ai = va[2 * i + 3];
        _br = vb[2 * i + 2];
        _bi = vb[2 * i + 3];
        VCPLXMUL(_ar, _ai, _br, _bi);
        vab[2 * i + 2] = VMADD(_ar, vscal, vab[2 * i + 2]);
        vab[2 * i + 3] = VMADD(_ai, vscal, vab[2 * i + 3]);
    }
#endif
    if (s->transform == PFFFT_REAL) {
        ((v4sf_union *) vab)[0].f[0] = abr + ar * br * scaling;
        ((v4sf_union *) vab)[1].f[0] = abi + ai * bi * scaling;
    }
}

#else // defined(PFFFT_SIMD_DISABLE)

// standard routine using scalar floats, without SIMD stuff.

#define pffft_zreorder_nosimd pffft_zreorder

void pffft_zreorder_nosimd(PFFFT_Setup *setup, const float *in, float *out, pffft_direction_t direction) {
    int k, N = setup->N;
    if (setup->transform == PFFFT_COMPLEX) {
        for (k = 0; k < 2 * N; ++k) out[k] = in[k];
        return;
    } else if (direction == PFFFT_FORWARD) {
        float x_N = in[N - 1];
        for (k = N - 1; k > 1; --k) out[k] = in[k - 1];
        out[0] = in[0];
        out[1] = x_N;
    } else {
        float x_N = in[1];
        for (k = 1; k < N - 1; ++k) out[k] = in[k + 1];
        out[0] = in[0];
        out[N - 1] = x_N;
    }
}

#define pffft_transform_internal_nosimd pffft_transform_internal

void pffft_transform_internal_nosimd(PFFFT_Setup *setup, const float *input, float *output, float *scratch,
                                     pffft_direction_t direction, int ordered) {
    size_t vec_size = setup->vec_size;
    int rad_odd = (setup->radix[1] & 1);

    // temporary buffer is allocated on the stack if the scratch pointer is NULL
    size_t stack_allocate = (scratch == 0 ? vec_size * 2 : 1);
    VLA_ARRAY_ON_STACK(v4sf, scratch_on_stack, stack_allocate);
    float *buff[2];
    int ib;
    if (scratch == 0) scratch = scratch_on_stack;
    buff[0] = output;
    buff[1] = scratch;

    if (setup->transform == PFFFT_COMPLEX) ordered = 0; // it is always ordered.
    ib = (rad_odd ^ ordered ? 1 : 0);

    if (direction == PFFFT_FORWARD) {
        if (setup->transform == PFFFT_REAL) {
            ib = (rfft_forward(vec_size * 2, input, buff[ib], buff[!ib],
                               setup->twiddle, &setup->radix[0]) == buff[0] ? 0 : 1);
        } else {
            ib = (cfft(vec_size, input, buff[ib], buff[!ib],
                       setup->twiddle, &setup->radix[0], -1) == buff[0] ? 0 : 1);
        }
        if (ordered) {
            pffft_zreorder(setup, buff[ib], buff[!ib], PFFFT_FORWARD);
            ib = !ib;
        }
    } else {
        if (input == buff[ib]) {
            ib = !ib; // may happen when finput == foutput
        }
        if (ordered) {
            pffft_zreorder(setup, input, buff[!ib], PFFFT_BACKWARD);
            input = buff[!ib];
        }
        if (setup->transform == PFFFT_REAL) {
            ib = (rfft_backward(vec_size * 2, input, buff[ib], buff[!ib],
                                setup->twiddle, &setup->radix[0]) == buff[0] ? 0 : 1);
        } else {
            ib = (cfft(vec_size, input, buff[ib], buff[!ib],
                       setup->twiddle, &setup->radix[0], +1) == buff[0] ? 0 : 1);
        }
    }
    if (buff[ib] != output) {
        int k;
        // extra copy required -- this situation should happens only when finput == foutput
        assert(input == output);
        for (k = 0; k < vec_size; ++k) {
            float a = buff[ib][2 * k], b = buff[ib][2 * k + 1];
            output[2 * k] = a;
            output[2 * k + 1] = b;
        }
        ib = !ib;
    }
    assert(buff[ib] == output);
}

#define pffft_zconvolve_accumulate_nosimd pffft_zconvolve_accumulate

void pffft_zconvolve_accumulate_nosimd(PFFFT_Setup *s, const float *a, const float *b,
                                       float *ab, float scaling) {
    size_t vec_size = s->vec_size;
    if (s->transform == PFFFT_REAL) {
        // take care of the fftpack ordering
        ab[0] += a[0] * b[0] * scaling;
        ab[2 * vec_size - 1] += a[2 * vec_size - 1] * b[2 * vec_size - 1] * scaling;
        ++ab;
        ++a;
        ++b;
        --vec_size;
    }
    int i;
    for (i = 0; i < vec_size; ++i) {
        float ar, ai, br, bi;
        ar = a[2 * i + 0];
        ai = a[2 * i + 1];
        br = b[2 * i + 0];
        bi = b[2 * i + 1];
        VCPLXMUL(ar, ai, br, bi);
        ab[2 * i + 0] += ar * scaling;
        ab[2 * i + 1] += ai * scaling;
    }
}

#endif // defined(PFFFT_SIMD_DISABLE)

void pffft_transform(PFFFT_Setup *setup, const float *input, float *output, float *work, pffft_direction_t direction) {
    pffft_transform_internal(setup, input, output, (v4sf *) work, direction, 0);
}

void pffft_transform_ordered(PFFFT_Setup *setup, const float *input, float *output, float *work,
                             pffft_direction_t direction) {
    pffft_transform_internal(setup, input, output, (v4sf *) work, direction, 1);
}

void fft_Bluestein_c2c(const cmplx *input, cmplx *output, size_t n, int flag) {
    int m = 1 << ((ilogbf((float) (2 * n - 1))));
    if (m < 2 * n - 1) {
        m <<= 1;
    }
    cmplx *y = (cmplx *) pffft_aligned_calloc(4 * m, sizeof(cmplx));
    PFFFT_Setup *fftctx = pffft_new_setup(m, PFFFT_COMPLEX);
    if (y == NULL || fftctx == NULL) {
        if (y)
            pffft_aligned_free(y);
        if (fftctx)
            pffft_destroy_setup(fftctx);
        return;
    }
    cmplx *w = y + m;
    cmplx *ww = w + m;
    cmplx *work = ww + m;
    w[0].real = 1;
    if (flag == -1) {
        ww[0].real = input[0].real;
        ww[0].imag = -input[0].imag;
        for (size_t i = 1; i < n; i++) {
            SinCos(M_PI * i * i / n, &w[i].imag, &w[i].real);
            w[m - i] = w[i];
            ww[i].real = (input[i].real * w[i].real) - (input[i].imag * w[i].imag);
            ww[i].imag = (-input[i].imag * w[i].real) - (input[i].real * w[i].imag);
        }
    } else {
        ww[0].real = input[0].real;
        ww[0].imag = input[0].imag;
        for (size_t i = 1; i < n; i++) {
            SinCos(M_PI * i * i / n, &w[i].imag, &w[i].real);
            w[m - i] = w[i];
            ww[i].real = (input[i].real * w[i].real) + (input[i].imag * w[i].imag);
            ww[i].imag = (input[i].imag * w[i].real) - (input[i].real * w[i].imag);
        }
    }
#ifdef PFFFT_SIMD_DISABLE
    pffft_transform(fftctx, (float *) ww, (float *) y, (float *) work, PFFFT_FORWARD);
 pffft_transform(fftctx, (float *) w, (float *) ww, (float *) work, PFFFT_FORWARD);
#else
    pffft_transform_ordered(fftctx, (float *) ww, (float *) y, (float *) work, PFFFT_FORWARD);
    pffft_transform_ordered(fftctx, (float *) w, (float *) ww, (float *) work, PFFFT_FORWARD);
#endif

    for (size_t i = 0; i < m; i++) {
        const float r = y[i].real;
        y[i].real = (r * ww[i].real) - (y[i].imag * ww[i].imag);
        y[i].imag = (y[i].imag * ww[i].real) + (r * ww[i].imag);
    }
#ifdef PFFFT_SIMD_DISABLE
    pffft_transform(fftctx, (float *) y, (float *) ww, (float *) work, PFFFT_BACKWARD);
#else
    pffft_transform_ordered(fftctx, (float *) y, (float *) ww, (float *) work, PFFFT_BACKWARD);
#endif
    if (flag == -1) {
        for (size_t i = 0; i < n; i++) {
            output[i].real = ((ww[i].real * w[i].real) + (ww[i].imag * w[i].imag)) / m;
            output[i].imag = -((ww[i].imag * w[i].real) - (ww[i].real * w[i].imag)) / m;
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            output[i].real = ((ww[i].real * w[i].real) + (ww[i].imag * w[i].imag)) / m;
            output[i].imag = ((ww[i].imag * w[i].real) - (ww[i].real * w[i].imag)) / m;
        }
    }
    pffft_aligned_free(y);
    pffft_destroy_setup(fftctx);
}

void fft_Bluestein_r2c(const float *input, cmplx *output, size_t n) {
    int m = 1 << ((ilogbf((float) (2 * n - 1))));
    if (m < 2 * n - 1) {
        m <<= 1;
    }
    cmplx *y = (cmplx *) pffft_aligned_calloc(4 * m, sizeof(cmplx));
    PFFFT_Setup *fftctx = pffft_new_setup(m, PFFFT_COMPLEX);

    if (y == NULL || fftctx == NULL) {
        if (y)
            pffft_aligned_free(y);
        if (fftctx)
            pffft_destroy_setup(fftctx);
        return;
    }
    cmplx *w = y + m;
    cmplx *ww = w + m;
    cmplx *work = ww + m;
    w[0].real = 1;
    ww[0].real = input[0];
    ww[0].imag = 0;
    for (size_t i = 1; i < n; i++) {
        SinCos(M_PI * i * i / n, &w[i].imag, &w[i].real);
        w[m - i] = w[i];
        ww[i].real = (input[i] * w[i].real);
        ww[i].imag = -(input[i] * w[i].imag);
    }
#ifdef PFFFT_SIMD_DISABLE
    pffft_transform(fftctx, (float *) ww, (float *) y, (float *) work, PFFFT_FORWARD);
 pffft_transform(fftctx, (float *) w, (float *) ww, (float *) work, PFFFT_FORWARD);
#else
    pffft_transform_ordered(fftctx, (float *) ww, (float *) y, (float *) work, PFFFT_FORWARD);
    pffft_transform_ordered(fftctx, (float *) w, (float *) ww, (float *) work, PFFFT_FORWARD);
#endif

    for (size_t i = 0; i < m; i++) {
        const float r = y[i].real;
        y[i].real = (r * ww[i].real) - (y[i].imag * ww[i].imag);
        y[i].imag = (y[i].imag * ww[i].real) + (r * ww[i].imag);
    }
#ifdef PFFFT_SIMD_DISABLE
    pffft_transform(fftctx, (float *) y, (float *) ww, (float *) work, PFFFT_BACKWARD);
#else
    pffft_transform_ordered(fftctx, (float *) y, (float *) ww, (float *) work, PFFFT_BACKWARD);
#endif
    for (size_t i = 0; i < n; i++) {
        output[i].real = ((ww[i].real * w[i].real) + (ww[i].imag * w[i].imag)) / m;
        output[i].imag = ((ww[i].imag * w[i].real) - (ww[i].real * w[i].imag)) / m;
    }
    pffft_aligned_free(y);
    pffft_destroy_setup(fftctx);
}

void fft_Bluestein_c2r(const cmplx *input, float *output, size_t n) {
    int m = 1 << ((ilogbf((float) (2 * n - 1))));
    if (m < 2 * n - 1) {
        m <<= 1;
    }
    cmplx *y = (cmplx *) pffft_aligned_calloc(4 * m, sizeof(cmplx));
    PFFFT_Setup *fftctx = pffft_new_setup(m, PFFFT_COMPLEX);

    if (y == NULL || fftctx == NULL) {
        if (y)
            pffft_aligned_free(y);
        if (fftctx)
            pffft_destroy_setup(fftctx);
        return;
    }
    cmplx *w = y + m;
    cmplx *ww = w + m;
    cmplx *work = ww + m;
    w[0].real = 1;
    ww[0].real = input[0].real;
    ww[0].imag = -input[0].imag;
    for (size_t i = 1; i < n; i++) {
        SinCos(M_PI * i * i / n, &w[i].imag, &w[i].real);
        w[m - i] = w[i];
        ww[i].real = (input[i].real * w[i].real) - (input[i].imag * w[i].imag);
        ww[i].imag = (-input[i].imag * w[i].real) - (input[i].real * w[i].imag);
    }
#ifdef PFFFT_SIMD_DISABLE
    pffft_transform(fftctx, (float *) ww, (float *) y, (float *) work, PFFFT_FORWARD);
 pffft_transform(fftctx, (float *) w, (float *) ww, (float *) work, PFFFT_FORWARD);
#else
    pffft_transform_ordered(fftctx, (float *) ww, (float *) y, (float *) work, PFFFT_FORWARD);
    pffft_transform_ordered(fftctx, (float *) w, (float *) ww, (float *) work, PFFFT_FORWARD);
#endif

    for (size_t i = 0; i < m; i++) {
        const float r = y[i].real;
        y[i].real = (r * ww[i].real) - (y[i].imag * ww[i].imag);
        y[i].imag = (y[i].imag * ww[i].real) + (r * ww[i].imag);
    }
#ifdef PFFFT_SIMD_DISABLE
    pffft_transform(fftctx, (float *) y, (float *) ww, (float *) work, PFFFT_BACKWARD);
#else
    pffft_transform_ordered(fftctx, (float *) y, (float *) ww, (float *) work, PFFFT_BACKWARD);
#endif
    for (size_t i = 0; i < n; i++) {
        output[i] = ((ww[i].real * w[i].real) + (ww[i].imag * w[i].imag)) / m;
    }
    pffft_aligned_free(y);
    pffft_destroy_setup(fftctx);
}

void FFT(const cmplx *input, cmplx *output, int n) {
    if (n < 2) {
        output[0] = input[0];
        return;
    }
    if (is_valid_size(n) == -1) {
        fft_Bluestein_c2c(input, output, n, 1);
    } else {
        PFFFT_Setup *fftctx = pffft_new_setup(n, PFFFT_COMPLEX);
        float *work = (float *) pffft_aligned_calloc(2 * n, sizeof(float));
        if (work == NULL || fftctx == NULL) {
            if (work)
                pffft_aligned_free(work);
            if (fftctx)
                pffft_destroy_setup(fftctx);
            return;
        }
        pffft_transform(fftctx, (float *) input, (float *) output, work, PFFFT_FORWARD);
        pffft_aligned_free(work);
        pffft_destroy_setup(fftctx);
    }
}

void IFFT(const cmplx *input, cmplx *output, int n) {
    if (n < 2) {
        output[0] = input[0];
        return;
    }
    if (is_valid_size(n) == -1) {
        fft_Bluestein_c2c(input, output, n, -1);
    } else {
        PFFFT_Setup *fftctx = pffft_new_setup(n, PFFFT_COMPLEX);
        float *work = (float *) pffft_aligned_calloc(2 * n, sizeof(float));
        if (work == NULL || fftctx == NULL) {
            if (work)
                pffft_aligned_free(work);
            if (fftctx)
                pffft_destroy_setup(fftctx);
            return;
        }
        pffft_transform(fftctx, (float *) input, (float *) output, work, PFFFT_BACKWARD);
        pffft_aligned_free(work);
        pffft_destroy_setup(fftctx);
    }
}

void FFT_r2c(const float *input, cmplx *output, int n) {
    if (n < 2) {
        output[0].real = input[0];
        return;
    }
    if (is_valid_size(n) == -1) {
        fft_Bluestein_r2c(input, output, n);
    } else {
        PFFFT_Setup *fftctx = pffft_new_setup(n, PFFFT_COMPLEX);
        float *work = (float *) pffft_aligned_calloc(4 * n, sizeof(float));
        if (work == NULL || fftctx == NULL) {
            if (work)
                pffft_aligned_free(work);
            if (fftctx)
                pffft_destroy_setup(fftctx);
            return;
        }
        cmplx *tmp = (cmplx *) (work + 2 * n);
        for (int i = 0; i < n; ++i) {
            tmp[i].real = input[i];
        }
        pffft_transform(fftctx, (float *) tmp, (float *) output, work, PFFFT_FORWARD);
        pffft_aligned_free(work);
        pffft_destroy_setup(fftctx);
    }
}

void IFFT_c2r(const cmplx *input, float *output, int n) {
    if (n < 2) {
        output[0] = input[0].real;
        return;
    }
    if (is_valid_size(n) == -1) {
        fft_Bluestein_c2r(input, output, n);
    } else {
        PFFFT_Setup *fftctx = pffft_new_setup(n, PFFFT_COMPLEX);
        float *work = (float *) pffft_aligned_calloc(4 * n, sizeof(float));
        if (work == NULL || fftctx == NULL) {
            if (work)
                pffft_aligned_free(work);
            if (fftctx)
                pffft_destroy_setup(fftctx);
            return;
        }
        float *tmp = work + (2 * n);
        pffft_transform(fftctx, (float *) input, tmp, work, PFFFT_BACKWARD);
        for (int i = 0; i < n; ++i) {
            output[i] = tmp[0];
            tmp += 2;
        }
        pffft_aligned_free(work);
        pffft_destroy_setup(fftctx);
    }
}