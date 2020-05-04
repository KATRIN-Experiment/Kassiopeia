/*
 * KSGenPositionSurfaceRandom.h
 *
 *  Created on: 17.09.2014
 *      Author: Jan Behrens
 */

#ifndef KSGENPOSITIONSURFACERANDOM_H_
#define KSGENPOSITIONSURFACERANDOM_H_

#include "KGCore.hh"
#include "KGRandomPointGenerator.hh"
#include "KSGenCreator.h"
#include "KSGenValue.h"
#include "KSGeneratorsMessage.h"

#include <vector>

namespace Kassiopeia
{
/**
    * \brief Dices positions of particles on surfaces.
    */
class KSGenPositionSurfaceRandom : public KSComponentTemplate<KSGenPositionSurfaceRandom, KSGenCreator>
{
  public:
    KSGenPositionSurfaceRandom();
    KSGenPositionSurfaceRandom(const KSGenPositionSurfaceRandom&);
    KSGenPositionSurfaceRandom* Clone() const override;
    ~KSGenPositionSurfaceRandom() override;

  public:
    /**
        * \brief Dices the positions of all particles of
        * the KSParticleQueue on surfaces which are
        * defined with AddSurface.
        *
        * \param aPrimaries
        */
    void Dice(KSParticleQueue* aPrimaries) override;

  public:
    /**
        * \brief Adds surfaces to the class in which the
        * position of the particles will be diced.
        *
        * \param aSurface
        */
    void AddSurface(KGeoBag::KGSurface* aSurface);

    /**
        * \brief Removes a surface from this class.
        *
        * \param aSurface
        */
    bool RemoveSurface(KGeoBag::KGSurface* aSurface);

  private:
    std::vector<KGeoBag::KGSurface*> fSurfaces;
    KGeoBag::KGRandomPointGenerator random;

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};
}  // namespace Kassiopeia

#endif /* KSGENPOSITIONSURFACERANDOM_H_ */
