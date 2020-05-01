#ifndef KGRANDOMPOINTGENERATOR_DEF
#define KGRANDOMPOINTGENERATOR_DEF

#include "KGBoxSpaceRandom.hh"
#include "KGBoxSurfaceRandom.hh"
#include "KGConeSpaceRandom.hh"
#include "KGConeSurfaceRandom.hh"
#include "KGCutConeSpaceRandom.hh"
#include "KGCutConeSurfaceRandom.hh"
#include "KGCutConeTubeSpaceRandom.hh"
#include "KGCylinderSpaceRandom.hh"
#include "KGCylinderSurfaceRandom.hh"
#include "KGDiskSurfaceRandom.hh"
#include "KGGenericSpaceRandom.hh"
#include "KGGenericSurfaceRandom.hh"
#include "KGRotatedSurfaceRandom.hh"

namespace KGeoBag
{
/**
   * \brief Main class for dicing random points
   * inside spaces and on surfaces.
   *
   * \detail There is a class KGGeneric???Random for calculation
   * of random points inside arbitrary spaces or on arbitrary surfaces.
   * The other classes implements specialized functions for these
   * calculations.
   */
class KGRandomPointGenerator :
    public KGGenericSpaceRandom,
    public KGGenericSurfaceRandom,
    // -------------------------------
    public KGRotatedSurfaceRandom,
    // -------------------------------
    public KGBoxSurfaceRandom,
    public KGBoxSpaceRandom,
    // -------------------------------
    public KGDiskSurfaceRandom,
    // -------------------------------
    public KGCylinderSpaceRandom,
    public KGCylinderSurfaceRandom,
    // -------------------------------
    public KGConeSpaceRandom,
    public KGConeSurfaceRandom,
    // -------------------------------
    public KGCutConeSpaceRandom,
    public KGCutConeSurfaceRandom,
    // -------------------------------
    public KGCutConeTubeSpaceRandom
{
  public:
    KGRandomPointGenerator() {}
    ~KGRandomPointGenerator() override {}
};
}  // namespace KGeoBag

#endif /* KGRANDOMPOINTGENERATOR_DEF */
