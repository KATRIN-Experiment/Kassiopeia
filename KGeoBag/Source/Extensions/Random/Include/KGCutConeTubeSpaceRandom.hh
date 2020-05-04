/*
 * KGCutConeTubeSpaceRandom.hh
 *
 *  Created on: 16.09.2015
 *      Author: Daniel Hilk
 */

#ifndef KGCUTCONETUBESPACERANDOM_HH_
#define KGCUTCONETUBESPACERANDOM_HH_

#include "KGCutConeTubeSpace.hh"
#include "KGShapeRandom.hh"

namespace KGeoBag
{
/**
   * \brief Class for dicing a point
   * inside a KGCutConeTubeSpace.
   */
class KGCutConeTubeSpaceRandom : virtual public KGShapeRandom, public KGCutConeTubeSpace::Visitor
{
  public:
    KGCutConeTubeSpaceRandom() : KGShapeRandom() {}
    ~KGCutConeTubeSpaceRandom() override {}

    /**
     * \brief Visitor function for dicing the point
     * inside the KGCutConeTubeSpace.
     *
     * \param aCutConeTubeSpace
     */
    void VisitCutConeTubeSpace(KGCutConeTubeSpace* aCutConeTubeSpace) override;

  private:
    double LinearInterpolation(double zInput, const double z1, const double r1, const double z2, const double r2);
};
}  // namespace KGeoBag

#endif /* KGCUTCONETUBESPACERANDOM_HH_ */
