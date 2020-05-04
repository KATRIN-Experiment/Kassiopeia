/**
 * @file KTRandom.h
 *
 * @date 27.11.2013
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */
#ifndef KTRANDOM_H_
#define KTRANDOM_H_

#include "KRandom.h"

#include <TRandom.h>

namespace katrin
{

class KTRandom : public TRandom, public KSingleton<KTRandom>
{
  public:
    KTRandom();
    ~KTRandom() override;

    UInt_t GetSeed() const override;
    Double_t Rndm(Int_t = 0) override;
    void RndmArray(Int_t n, Float_t* array) override;
    void RndmArray(Int_t n, double* array) override;
    void SetSeed(ULong_t seed = 0) override;

  private:
    KRandom& fGenerator;

    //    ClassDef(KTRandom, 0);
};

} /* namespace katrin */

R__EXTERN TRandom* gRandom;

#endif /* KTRANDOM_H_ */
