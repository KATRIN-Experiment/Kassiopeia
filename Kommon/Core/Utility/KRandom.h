/**
 * @file KRandom.h
 *
 * @date 24.11.2013
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 *
 *
 * Original C-program for MT19937 coded by Takuji Nishimura and Makoto Matsumoto.
 *
 * Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *   3. The names of its contributors may not be used to endorse or promote
 *      products derived from this software without specific prior written
 *      permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Any feedback is very welcome.
 * http://www.math.keio.ac.jp/matumoto/emt.html
 * email: matumoto@math.keio.ac.jp
 *
 *
 * C++ codes by Kohei Takeda (k-tak@letter.or.jp)
 * Redistribution terms are the same as the original ones above.
 *
 *
 * Ingration into the KATRIN simulation and analysis framework by Marco Kleesiek <marco.kleesiek@kit.edu>.
 *
 */

#ifndef KRANDOM_H_
#define KRANDOM_H_

#include "KSingleton.h"
#include "KNonCopyable.h"

#include <cmath>
#include <cstdlib>
#include <cassert>
#include <stdint.h>
#include <limits>

namespace katrin {

namespace detail {

/**
 * 32 bit traits for the mersenne twister algorithm.
 */
struct mt19937_32_traits
{
    typedef unsigned int UINTTYPE;
    typedef signed int INTTYPE;
    static const int INTTYPE_BITS = 32;
    static const unsigned int MAXDOUBLEVAL = 4294967295U; //2^32-1
    static const size_t NN = 624;
    static const size_t MM = 397;
    static const unsigned int INITVAL = 1812433253U;
    static const unsigned int ARRAYINITVAL_0 = 19650218U;
    static const unsigned int ARRAYINITVAL_1 = 1664525U;
    static const unsigned int ARRAYINITVAL_2 = 1566083941U;

    static unsigned int twist(const unsigned int& u, const unsigned int& v)
    {
        static unsigned int mag01[2] = { 0U, 0x9908b0dfU };
        return ((((u & 0x80000000U) | (v & 0x7fffffffU)) >> 1) ^ mag01[v & 1]);
    }

    static unsigned int temper(unsigned int y)
    {
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);

        return y;
    }

    static double real1(unsigned int y)
    {
        return ((double) temper(y) * (1.0 / (double) MAXDOUBLEVAL));
    }

    static double real2(unsigned int y)
    {
        return ((double) temper(y) * (1.0 / ((double) MAXDOUBLEVAL + 1.0)));
    }

    static double real3(unsigned int y)
    {
        return (((double) temper(y) + 0.5) * (1.0 / ((double) MAXDOUBLEVAL + 1.0)));
    }
};

/**
 * 64 bit traits for the mersenne twister algorithm.
 */
struct mt19937_64_traits
{
    typedef unsigned long long UINTTYPE;
    typedef signed long long INTTYPE;
    static const int INTTYPE_BITS = 64;
    static const unsigned long long MAXDOUBLEVAL = 9007199254740991ULL; // 2^53-1
    static const size_t NN = 312;
    static const size_t MM = 156;
    static const unsigned long long INITVAL = 6364136223846793005ULL;
    static const unsigned long long ARRAYINITVAL_0 = 19650218ULL;
    static const unsigned long long ARRAYINITVAL_1 = 3935559000370003845ULL;
    static const unsigned long long ARRAYINITVAL_2 = 2862933555777941757ULL;

    static unsigned long long twist(const unsigned long long& u, const unsigned long long& v)
    {
        static unsigned long long mag01[2] = { 0ULL, 0xB5026F5AA96619E9ULL };
        return ((((u & 0xFFFFFFFF80000000ULL) | (v & 0x7FFFFFFFULL)) >> 1) ^ mag01[v & 1]);
    }

    static unsigned long long temper(unsigned long long y)
    {
        y ^= (y >> 29) & 0x5555555555555555ULL;
        y ^= (y << 17) & 0x71D67FFFEDA60000ULL;
        y ^= (y << 37) & 0xFFF7EEE000000000ULL;
        y ^= (y >> 43);

        return y;
    }

    static double real1(unsigned int y)
    {
        return ((double) (temper(y) >> (INTTYPE_BITS - 53)) * (1.0 / 9007199254740991.0));
    }

    static double real2(unsigned int y)
    {
        return ((double) (temper(y) >> (INTTYPE_BITS - 53)) * (1.0 / 9007199254740992.0));
    }

    static double real3(unsigned int y)
    {
        return (((double) (temper(y) >> (INTTYPE_BITS - 52)) + 0.5) * (1.0 / 4503599627370496.0));
    }
};

/**
 * Mersenne Twister algorithm implementation.
 */
template<typename Traits>
class mt19937_prototype
{
public:
    typedef typename Traits::UINTTYPE result_type;
    typedef typename Traits::INTTYPE signed_result_type;

    static const result_type skDefaultSeed = 5489u;

public:
    mt19937_prototype(result_type seedval = skDefaultSeed) :
        state_( (result_type*) malloc(sizeof(result_type) * Traits::NN) ),
        left_( 1 ),
        next_( 0 )
    {
        seed(seedval);
    }

    virtual ~mt19937_prototype()
    {
        if (state_)
            free(state_);
    }

    void seed(result_type seedval = skDefaultSeed)
    {
        assert(sizeof(result_type)*8 == (size_t) Traits::INTTYPE_BITS);

        state_[0] = seedval;
        for (size_t j = 1; j < Traits::NN; j++) {
            state_[j] = (Traits::INITVAL * (state_[j - 1] ^ (state_[j - 1] >> (Traits::INTTYPE_BITS - 2)))
                    + (result_type) j);
        }
        left_ = 1;
    }

    result_type min() const
    {
        return 0;
    }

    result_type max() const
    {
        return Traits::MAXDOUBLEVAL;
    }

    result_type operator()()
    {
        return getUint();
    }

    void discard(result_type z = 1)
    {
        for (result_type i=0; i < z; ++i)
            nextState();
    }

protected:
    mt19937_prototype(result_type initkeys[], size_t keylen) :
        state_( (result_type*) malloc(sizeof(result_type) * Traits::NN) ),
        left_( 1 ),
        next_( 0 )
    {
        seed(initkeys, keylen);
    }

    void seed(result_type initkeys[], size_t keylen)
    {
        seed(Traits::ARRAYINITVAL_0);

        size_t i = 1;
        size_t j = 0;
        size_t k = (Traits::NN > keylen ? Traits::NN : keylen);

        for (; k; k--) {
            state_[i] = (state_[i]
                    ^ ((state_[i - 1] ^ (state_[i - 1] >> (Traits::INTTYPE_BITS - 2)))
                            * Traits::ARRAYINITVAL_1)) + initkeys[j] + (result_type) j; /* non linear */

            i++;
            j++;

            if (i >= Traits::NN) {
                state_[0] = state_[Traits::NN - 1];
                i = 1;
            }
            if (j >= keylen) {
                j = 0;
            }
        }

        for (k = Traits::NN - 1; k; k--) {
            state_[i] = (state_[i]
                    ^ ((state_[i - 1] ^ (state_[i - 1] >> (Traits::INTTYPE_BITS - 2)))
                            * Traits::ARRAYINITVAL_2)) - (result_type) i; /* non linear */

            i++;

            if (i >= Traits::NN) {
                state_[0] = state_[Traits::NN - 1];
                i = 1;
            }
        }

        /* MSB is 1; assuring non-zero initial array */
        state_[0] = (result_type) 1 << (Traits::INTTYPE_BITS - 1);
        left_ = 1;
    }

    /* generates a random number on [0,2^bits-1]-interval */
    result_type getUint()
    {
        if (--left_ == 0)
            nextState();
        return Traits::temper(*next_++);
    }

    /* generates a random number on [0,2^(bits-1)-1]-interval */
    signed_result_type getInt()
    {
        if (--left_ == 0)
            nextState();
        return (signed_result_type) (Traits::temper(*next_++) >> 1);
    }

    /* generates a random number on [0,1]-real-interval */
    double getReal1()
    {
        if (--left_ == 0)
            nextState();
        return Traits::real1(*next_++);
    }

    /* generates a random number on [0,1)-real-interval */
    double getReal2()
    {
        if (--left_ == 0)
            nextState();
        return Traits::real2(*next_++);
    }

    /* generates a random number on (0,1)-real-interval */
    double getReal3()
    {
        if (--left_ == 0)
            nextState();
        return Traits::real3(*next_++);
    }

    void nextState()
    {
        result_type *p = state_;
        size_t j;

        left_ = Traits::NN;
        next_ = state_;

        for (j = Traits::NN - Traits::MM + 1; --j; p++)
            *p = p[Traits::MM] ^ Traits::twist(p[0], p[1]);

        for (j = Traits::MM; --j; p++)
            *p = p[Traits::MM - Traits::NN] ^ Traits::twist(p[0], p[1]);

        *p = p[Traits::MM - Traits::NN] ^ Traits::twist(p[0], state_[0]);
    }

protected:
    result_type* state_;
    size_t left_;
    result_type* next_;
};

/// The 32 bit mersenne twister. Used by default within the KATRIN framework on all deployed architectures.
typedef mt19937_prototype<mt19937_32_traits> mt19937;

/// Optional 64 bit mersenne twister.
typedef mt19937_prototype<mt19937_64_traits> mt19937_64;

} /* namespace detail */


/**
 * A Mersenne Twister random number generator, which should be used as a singleton.
 * This class implements the STL C++11 random number engine interface, so it can be extended according to your needs.
 */
class KRandom : public detail::mt19937, public KSingletonAsReference<KRandom>, KNonCopyable
{
public:
    KRandom(result_type seed = detail::mt19937::skDefaultSeed);
    virtual ~KRandom();

    result_type GetSeed() const;

    /**
     * Set the seed on the underlying mersenne twister engine.
     * For a seed = 0, the current system time in seconds is used as seed value.
     * @param seed
     * @return
     */
    result_type SetSeed(result_type seed = detail::mt19937::skDefaultSeed);

    /**
     * Get a random uniform number in the specified range.
     * By default the interval [min, max) is used.
     * @param min lower interval bound
     * @param max upper interval bound
     * @param minIncluded Make the lower bound closed.
     * @param maxIncluded Make the upper bound closed.
     *
     * @return
     */
    double Uniform(double min = 0.0, double max = 1.0, bool minIncluded = true, bool maxIncluded = false);

    /**
     * Get a random uniform number from a discrete distribution [inclMin, inclMax].
     * @param inclMin
     * @param inclMax
     * @return
     */
    int32_t Uniform(int32_t inclMin, int32_t inclMax);

    int64_t Uniform(int64_t inclMin, int64_t inclMax);

    /**
     * Return a boolean.
     * @param probability
     * @return True with the given probability.
     */
    bool Bool(double probability);

    /**
     * Draw from a gaussian distribution.
     * Uses the Acceptance-complement ratio from W. Hoermann and G. Derflinger
     * @param mean
     * @param sigma
     * @return
     * @see   W. Hoermann and G. Derflinger (1990):
     *        The ACR Method for generating normal random variables, OR Spektrum 12 (1990), 181-185.
     */
    double Gauss(double mean = 0.0, double sigma = 1.0);

    /**
     * Draw from an exponential distribution according to exp(-t/tau).
     * @param tau
     * @return
     */
    double Exponential(double tau);

    /**
     * Draw from a poissonian distribution.
     * Uses the PTRD algorithm from W. Hoermann.
     * @param mean
     * @return
     * @see "The transformed rejection method for generating Poisson random variables", Wolfgang Hormann,
     *       Insurance: Mathematics and Economics Volume 12, Issue 1, February 1993, Pages 39-45
     */
    uint32_t Poisson(double mean);

    double PoissonDouble(double mean);

private:
    KRandom(const KRandom& other);
    KRandom& operator= (const KRandom& other);

    result_type fSeed;
};

inline KRandom::KRandom(result_type seed) :
        detail::mt19937(),
        fSeed(0)
{
    SetSeed(seed);
}

inline KRandom::~KRandom()
{ }

inline KRandom::result_type KRandom::GetSeed() const
{
    return fSeed;
}

inline double KRandom::Uniform(double min, double max, bool minIncluded, bool maxIncluded)
{
    if (minIncluded)
        if (!maxIncluded)
            return min + getReal2() * (max - min);
        else
            return min + getReal1() * (max - min);
    else
        if (!maxIncluded)
            return min + getReal3() * (max - min);
        else
            return min + (1.0 - getReal2()) * (max - min);
}

inline int32_t KRandom::Uniform(int32_t inclMin, int32_t inclMax)
{
    return inclMin + getUint() % (inclMax - inclMin + 1);
}

inline int64_t KRandom::Uniform(int64_t inclMin, int64_t inclMax)
{
    return inclMin + getUint() % (inclMax - inclMin + 1);
}

inline bool KRandom::Bool(double probability)
{
    return (probability >= 1.0 || getReal2() < probability);
}

} /* namespace katrin */

#endif /* KRANDOM_H_ */
