/*
 * KSGenPositionSpaceRandom.h
 *
 *  Created on: 21.05.2014
 *      Author: Jan Oertlin
 */

#ifndef KSGENPOSITIONSPACERANDOM_H_
#define KSGENPOSITIONSPACERANDOM_H_

#include "KGCore.hh"
#include "KGRandomPointGenerator.hh"
#include "KSGenCreator.h"
#include "KSGenValue.h"
#include "KSGeneratorsMessage.h"

#include <vector>

namespace Kassiopeia
{
/**
         * \brief Dices positions of particles inside of spaces.
         */
class KSGenPositionSpaceRandom : public KSComponentTemplate<KSGenPositionSpaceRandom, KSGenCreator>
{
  public:
    KSGenPositionSpaceRandom();
    KSGenPositionSpaceRandom(const KSGenPositionSpaceRandom&);
    KSGenPositionSpaceRandom* Clone() const override;
    ~KSGenPositionSpaceRandom() override;

  public:
    /**
             * \brief Dices the positions of all particles of
             * the KSParticleQueue inside of spaces whiche are
             * defined with AddSpace.
             *
             * \param aPrimaries
             */
    void Dice(KSParticleQueue* aPrimaries) override;

  public:
    /**
             * \brief Adds spaces to the class in which the
             * position of the particles will be diced.
             *
             * \param aSpace
             */
    void AddSpace(KGeoBag::KGSpace* aSpace);

    /**
             * \brief Removes a space from this class.
             *
             * \param aSpace
             */
    bool RemoveSpace(KGeoBag::KGSpace* aSpace);

  private:
    std::vector<KGeoBag::KGSpace*> fSpaces;
    KGeoBag::KGRandomPointGenerator random;

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};
}  // namespace Kassiopeia

#endif /* KSGENPOSITIONSPACE_H_ */
