/*
 * KSGenPositionMeshSurfaceRandom.h
 *
 *  Created on: 28.01.2015
 *      Author: Nikolaus Trost
 */

#ifndef _KSGenPositionMeshSurfaceRandom_H_
#define _KSGenPositionMeshSurfaceRandom_H_

#include "KGCore.hh"
#include "KGMesh.hh"
#include "KSGenCreator.h"
#include "KSGenValue.h"
#include "KSGeneratorsMessage.h"

#include <vector>

namespace Kassiopeia
{
/**
    * \brief Dices positions of particles on discretized surfaces.
    */
class KSGenPositionMeshSurfaceRandom :
    public KGeoBag::KGVisitor,
    public KGeoBag::KGSurface::Visitor,
    public KGeoBag::KGExtendedSurface<KGeoBag::KGMesh>::Visitor,
    public KSComponentTemplate<KSGenPositionMeshSurfaceRandom, KSGenCreator>
{

  public:
    KSGenPositionMeshSurfaceRandom();
    KSGenPositionMeshSurfaceRandom(const KSGenPositionMeshSurfaceRandom&);
    KSGenPositionMeshSurfaceRandom* Clone() const override;
    ~KSGenPositionMeshSurfaceRandom() override;

  public:
    /**
        * \brief Dices the positions of all particles of
        * the KSParticleQueue on meshed surfaces which are visited at
        * the creation of the Object
        *
        * @param aPrimaries
        */
    void Dice(KSParticleQueue* aPrimaries) override;

  public:
    /**
         * @brief Obtains the coordinate system of a surface and stores it
         * @param aSurface
         */
    void VisitSurface(KGeoBag::KGSurface* aSurface) override;

    /**
         * @brief Visits a (non-axial) Mesh of a Surface, retrieves all the mesh elements
         *  and stores them locally
         * @param aSurface
         */
    void VisitExtendedSurface(KGeoBag::KGExtendedSurface<KGeoBag::KGMesh>* aSurface) override;

  private:
    /**
         * @brief fTotalArea - total area of the visited surfaces.
         * needed for dicing the surface element the position will be on
         */
    double fTotalArea;

    /**
         * @brief The KSGenCoordinatesystem struct will hold the transformations between the
         * gloabal coordinate system and the local ones belonging to the surfaces
         */
    struct KSGenCoordinatesystem
    {
        KGeoBag::KThreeVector fOrigin;
        KGeoBag::KThreeVector fXAxis;
        KGeoBag::KThreeVector fYAxis;
        KGeoBag::KThreeVector fZAxis;
    };

    /**
         * @brief KSGenMeshElementSystem is the unit of a meshed surface and its coordinate system
         */
    typedef std::pair<KSGenCoordinatesystem, KGeoBag::KGMeshElementVector*> KSGenMeshElementSystem;

    /**
         * @brief fElementsystems holds all the Elementsystems of the different
         * surfaces added during the set up of the object
         */
    std::vector<KSGenMeshElementSystem> fElementsystems;


  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
};
}  // namespace Kassiopeia


#endif /*_KSGenPositionMeshSurfaceRandom_H_*/
