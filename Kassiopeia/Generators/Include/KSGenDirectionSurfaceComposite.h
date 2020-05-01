/*
 * KSGenDirectionSurfaceComposite.h
 *
 *  Created on: 17.09.2014
 *      Author: J. Behrens
 */

#ifndef KSGENDIRECTIONSURFACECOMPOSITE_H_
#define KSGENDIRECTIONSURFACECOMPOSITE_H_

#include "KGCore.hh"
#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

class KSGenDirectionSurfaceComposite : public KSComponentTemplate<KSGenDirectionSurfaceComposite, KSGenCreator>
{
  public:
    KSGenDirectionSurfaceComposite();
    KSGenDirectionSurfaceComposite(const KSGenDirectionSurfaceComposite& aCopy);
    KSGenDirectionSurfaceComposite* Clone() const override;
    ~KSGenDirectionSurfaceComposite() override;

  public:
    /**
        * \brief Dices the positions of all particles of
        * the KSParticleQueue on surfaces which are
        * defined with AddSurface.
        *
        * \param aPrimaries
        */
    void Dice(KSParticleQueue* aParticleList) override;

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

    void SetThetaValue(KSGenValue* anThetaValue);
    void ClearThetaValue(KSGenValue* anThetaValue);

    void SetPhiValue(KSGenValue* aPhiValue);
    void ClearPhiValue(KSGenValue* aPhiValue);

    void SetSide(bool aSide);

  private:
    std::vector<KGeoBag::KGSurface*> fSurfaces;

    KSGenValue* fThetaValue;
    KSGenValue* fPhiValue;

    bool fOutside;

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};

}  // namespace Kassiopeia

#endif /* KSGENDIRECTIONSURFACECOMPOSITE_H_ */
