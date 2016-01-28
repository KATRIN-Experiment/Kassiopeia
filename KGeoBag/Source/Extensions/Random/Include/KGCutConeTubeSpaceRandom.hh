/*
 * KGCutConeTubeSpaceRandom.hh
 *
 *  Created on: 16.09.2015
 *      Author: Daniel Hilk
 */

#ifndef KGCUTCONETUBESPACERANDOM_HH_
#define KGCUTCONETUBESPACERANDOM_HH_

#include "KGShapeRandom.hh"
#include "KGCutConeTubeSpace.hh"

namespace KGeoBag
{
  /**
   * \brief Class for dicing a point
   * inside a KGCutConeTubeSpace.
   */
  class KGCutConeTubeSpaceRandom : virtual public KGShapeRandom,
				 public KGCutConeTubeSpace::Visitor
  {
  public:
	  KGCutConeTubeSpaceRandom() : KGShapeRandom() {}
    virtual ~KGCutConeTubeSpaceRandom() {}

    /**
     * \brief Visitor function for dicing the point
     * inside the KGCutConeTubeSpace.
     *
     * \param aCutConeTubeSpace
     */
    virtual void VisitCutConeTubeSpace(KGCutConeTubeSpace* aCutConeTubeSpace);
  private:
    double LinearInterpolation( double zInput, const double z1, const double r1, const double z2, const double r2 );
  };
}

#endif /* KGCUTCONETUBESPACERANDOM_HH_ */
