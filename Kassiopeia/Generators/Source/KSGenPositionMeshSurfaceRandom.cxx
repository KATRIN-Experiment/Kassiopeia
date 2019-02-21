/*
 * KSGenPositionMeshSurfaceRandom.cxx
 *
 *  Created on: 28.01.2015
 *      Author: Nikolaus Trost
 */

#include "KSGenPositionMeshSurfaceRandom.h"
#include "KGCore.hh"
#include "KGMesh.hh"
#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KRandom.h"
using katrin::KRandom;

namespace Kassiopeia
{
    KSGenPositionMeshSurfaceRandom::KSGenPositionMeshSurfaceRandom():
            fTotalArea( 0. )
    {}
    KSGenPositionMeshSurfaceRandom::KSGenPositionMeshSurfaceRandom(const KSGenPositionMeshSurfaceRandom& aCopy):
            KSComponent(),
            fTotalArea( aCopy.fTotalArea ),
            fElementsystems( aCopy.fElementsystems )
    {}

    KSGenPositionMeshSurfaceRandom* KSGenPositionMeshSurfaceRandom::Clone() const
    {
        return new KSGenPositionMeshSurfaceRandom(*this);
    }

    KSGenPositionMeshSurfaceRandom::~KSGenPositionMeshSurfaceRandom() {}

    void KSGenPositionMeshSurfaceRandom::VisitSurface( KGeoBag::KGSurface* aSurface )
    {
        struct KSGenCoordinatesystem tCoordinateSystem = { aSurface->GetOrigin(),
                                                            aSurface->GetXAxis(),
                                                            aSurface->GetYAxis(),
                                                            aSurface->GetZAxis() };
        KSGenMeshElementSystem tSystem;
        tSystem.first = tCoordinateSystem;
        tSystem.second = NULL;

        fElementsystems.push_back( tSystem );

        return;
    }

    void KSGenPositionMeshSurfaceRandom::VisitExtendedSurface( KGeoBag::KGExtendedSurface< KGeoBag::KGMesh >* anExtendedSurface )
    {
        fElementsystems.back().second = anExtendedSurface->Elements();

        return;
    }

    void KSGenPositionMeshSurfaceRandom::Dice(KSParticleQueue* aPrimaries)
    {
        for(KSParticleIt tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); ++tParticleIt)
        {
            double tDecision = KRandom::GetInstance().Uniform() * fTotalArea;
            std::vector< KSGenMeshElementSystem >::iterator tSysIt;
            KGeoBag::KGMeshElementCIt tElementIt;

            for( tSysIt = fElementsystems.begin(); tSysIt != fElementsystems.end(); ++tSysIt )
            {
                for( tElementIt = tSysIt->second->begin(); tElementIt != tSysIt->second->end(); ++tElementIt )
                {
                    tDecision -= (*tElementIt)->Area();
                    if (tDecision < 0.)
                    {
                        break;
                    }
                }

                if( tDecision < 0.)
                    break;
            }

            if( KGeoBag::KGMeshRectangle* tMeshRectangle = dynamic_cast< KGeoBag::KGMeshRectangle* >( *tElementIt ) )
            {
                KThreeVector tInternalRandomPosition = tMeshRectangle->GetP0() + KRandom::GetInstance().Uniform() * tMeshRectangle->GetA() * tMeshRectangle->GetN1()
                                                                               + KRandom::GetInstance().Uniform() * tMeshRectangle->GetB() * tMeshRectangle->GetN2();

                KThreeVector tExternalRandomPosition = tSysIt->first.fOrigin + tInternalRandomPosition.X() * tSysIt->first.fXAxis
                                                                             + tInternalRandomPosition.Y() * tSysIt->first.fYAxis
                                                                             + tInternalRandomPosition.Z() * tSysIt->first.fZAxis;

                (*tParticleIt)->SetPosition(tExternalRandomPosition);
                genmsg( eDebug ) << "surface random position generator <" << GetName() << "> diced position " << tExternalRandomPosition << " on rectangle" << eom;
                continue;
            }

            if( KGeoBag::KGMeshTriangle* tMeshTriangle = dynamic_cast< KGeoBag::KGMeshTriangle* >( *tElementIt ) )
            {
                //P = (1 - sqrt(r1)) * A + (sqrt(r1) * (1 - r2)) * B + (sqrt(r1) * r2) * C
                double r1 = KRandom::GetInstance().Uniform();
                double r2 = KRandom::GetInstance().Uniform();

                KThreeVector tInternalRandomPosition = (1 - sqrt(r1)) * tMeshTriangle->GetP0()
                                                    + (sqrt(r1) * (1 - r2)) * tMeshTriangle->GetP1()
                                                    + (sqrt(r1) * r2) * tMeshTriangle->GetP2();

                KThreeVector tExternalRandomPosition = tSysIt->first.fOrigin + tInternalRandomPosition.X() * tSysIt->first.fXAxis
                                                                             + tInternalRandomPosition.Y() * tSysIt->first.fYAxis
                                                                             + tInternalRandomPosition.Z() * tSysIt->first.fZAxis;

                (*tParticleIt)->SetPosition(tExternalRandomPosition);
                genmsg( eDebug ) << "surface random position generator <" << GetName() << "> diced position " << tExternalRandomPosition << " on triangle" << eom;
                continue;

            }

            if( KGeoBag::KGMeshWire* tMeshWire = dynamic_cast< KGeoBag::KGMeshWire* >( *tElementIt ) )
            {
                KThreeVector tStart = tMeshWire->GetP1();
                KThreeVector tEnd = tMeshWire->GetP0();
                KThreeVector tConnection = tEnd - tStart;
                KThreeVector tOrthogonal = tConnection.Orthogonal();
                KThreeVector tU = tOrthogonal.Unit();
                KThreeVector tV = tConnection.Cross( tOrthogonal ).Unit();

                double tRadius = tMeshWire->GetDiameter() / 2.;
                double tPhi = KRandom::GetInstance().Uniform()*360.;

                KThreeVector tInternalRandomPosition = tStart + KRandom::GetInstance().Uniform() * tConnection
                                                              + tRadius * (cos( tPhi ) * tU + sin( tPhi ) * tV);

                KThreeVector tExternalRandomPosition = tSysIt->first.fOrigin + tInternalRandomPosition.X() * tSysIt->first.fXAxis
                                                                             + tInternalRandomPosition.Y() * tSysIt->first.fYAxis
                                                                             + tInternalRandomPosition.Z() * tSysIt->first.fZAxis;

                (*tParticleIt)->SetPosition(tExternalRandomPosition);
                genmsg( eDebug ) << "surface random position generator <" << GetName() << "> diced position " << tExternalRandomPosition << " on wire" << eom;
                continue;
            }
            genmsg( eError ) << "Could not dice a position! Only rectangles, triangles and wires are acceptable elements. " << eom;
        }
    }

    void KSGenPositionMeshSurfaceRandom::InitializeComponent()
    {

        for( std::vector< KSGenMeshElementSystem >::iterator tSysIt = fElementsystems.begin();
                                                              tSysIt != fElementsystems.end();
                                                              ++tSysIt )
        {
            if ((tSysIt->second)==NULL) {
                genmsg( eError ) << "Mesh has not been defined for all surfaces specified in KSGenPositionMeshSurfaceRandom" << eom;
            }
            
            for( KGeoBag::KGMeshElementCIt tElementIt = tSysIt->second->begin();
                                          tElementIt != tSysIt->second->end();
                                          ++tElementIt )
            {
                fTotalArea += (*tElementIt)->Area();
            }
        }

        if( fTotalArea <= 0. || fTotalArea != fTotalArea )
            genmsg(eError) << "KSGenPositionRandomSurface fTotalArea is " << fTotalArea << " wrong/corrupt mesh?" << eom;

        genmsg( eDebug ) << "KSGenPositionRandomSurface fTotalArea: " << fTotalArea << eom;
        return;
    }
    void KSGenPositionMeshSurfaceRandom::DeinitializeComponent()
    {}

}
