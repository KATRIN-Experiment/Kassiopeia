/**
 * @file KTRandom.cxx
 *
 * @date 27.11.2013
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */
#include "KTRandom.h"

#include "KRandom.h"

TRandom* gRandom = katrin::KTRandom::GetInstance();

//ClassImp(katrin::KTRandom)

namespace katrin
{

KTRandom::KTRandom() :
        fGenerator( KRandom::GetInstance() )
{
    SetName("KTRandom");
    SetTitle("Random number generator: KATRIN's Mersenne Twistor");
}

KTRandom::~KTRandom()
{ }

UInt_t KTRandom::GetSeed() const
{
    return fGenerator.GetSeed();
}

Double_t KTRandom::Rndm(Int_t)
{
    return fGenerator.Uniform(0.0, 1.0, false, false);
}

void KTRandom::RndmArray(Int_t n, Float_t *array)
{
    for (UInt_t i = 0; i < (UInt_t) n; ++i)
        *(array + i) = fGenerator.Uniform(0.0, 1.0, false, true);
}

void KTRandom::RndmArray(Int_t n, double *array)
{
    for (UInt_t i = 0; i < (UInt_t) n; ++i)
        *(array + i) = fGenerator.Uniform(0.0, 1.0, false, true);
}

void KTRandom::SetSeed(UInt_t seed)
{
    fGenerator.SetSeed(seed);
}

} /* namespace katrin */
