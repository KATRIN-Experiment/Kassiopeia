/**
 * @file KRandomRootInterface.h
 *
 * @date 24.01.2022
 * @author Jan Behrens <jan.behrens@kit.edu>
 *
 */

#ifndef KRANDOM_ROOT_INTERFACE_H_
#define KRANDOM_ROOT_INTERFACE_H_

#include <KRandom.h>
#include <TRandom.h>

namespace katrin
{

namespace Kommon
{

template<class XEngineType = KRandom::engine_type>
class KRandomRootInterface :
    public TRandom
{
public:
        KRandomRootInterface(RandomPrototype<XEngineType>& generator = KRandom::GetInstance()) :
            fGenerator(generator)
        {}

        virtual ~KRandomRootInterface() = default;

public:
    virtual UInt_t GetSeed() const;
    virtual void SetSeed(ULong_t seed=0);
    virtual Double_t Rndm();
    // keep for backward compatibility
    virtual Double_t Rndm(Int_t);
    virtual void RndmArray(Int_t, Double_t*);
    virtual void RndmArray(Int_t, Float_t*);

private:
    RandomPrototype<XEngineType>& fGenerator;
};

template<class XEngineType>
UInt_t KRandomRootInterface<XEngineType>::GetSeed() const
{
    return fGenerator.GetSeed();
}

template<class XEngineType>
void KRandomRootInterface<XEngineType>::SetSeed(ULong_t seed)
{
    fGenerator.SetSeed(static_cast<typename XEngineType::result_type>(seed));
}

template<class XEngineType>
Double_t KRandomRootInterface<XEngineType>::Rndm()
{
    return fGenerator.Uniform(0., 1., false, false);
}

template<class XEngineType>
Double_t KRandomRootInterface<XEngineType>::Rndm(Int_t)
{
    return Rndm();
}

template<class XEngineType>
void KRandomRootInterface<XEngineType>::RndmArray(Int_t n, Double_t *array)
{
    Int_t k = 0;
    while (k < n) {
        array[k] = fGenerator.Uniform(0., 1., false, false);
        k++;
    }
}

template<class XEngineType>
void KRandomRootInterface<XEngineType>::RndmArray(Int_t n, Float_t *array)
{
    Int_t k = 0;
    while (k < n) {
        array[k] = fGenerator.Uniform(0., 1., false, false);
        k++;
    }
}

} /*namespace Kommon */

} /* namespace katrin */

#endif /* KRANDOM_ROOT_INTERFACE_H_ */
