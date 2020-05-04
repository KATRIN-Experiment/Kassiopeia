/*
 * KGBoxSurfaceRandom.hh
 *
 *  Created on: 12.05.2014
 *      Author: oertlin
 */

#ifndef KGBOXSURFACERANDOM_HH_
#define KGBOXSURFACERANDOM_HH_

#include "KGBox.hh"
#include "KGShapeRandom.hh"

namespace KGeoBag
{
/**
   * \brief Class for dicing a point on a KGBoxSurface.
   */
class KGBoxSurfaceRandom : virtual public KGShapeRandom, public KGBox::Visitor
{
  public:
    KGBoxSurfaceRandom() : KGShapeRandom() {}
    ~KGBoxSurfaceRandom() override {}

    /**
     * \brief Visitor function for dicing a point on
     * a KGBox.
     *
     * \brief aBox
     */
    void VisitBox(KGBox* aBox) override;
};
}  // namespace KGeoBag

#endif /* KGBOXSURFACERANDOM_HH_ */
