#include "KGROOTGeometryPainter.hh"
#include "KGVisualizationMessage.hh"

#include "KFile.h"
using katrin::KFile;

#include "KConst.h"
using katrin::KConst;

#include <TColor.h>

#include <cmath>
#include <limits>

using namespace std;

namespace KGeoBag
{

	KGROOTGeometryPainter::KGROOTGeometryPainter() :
            fDefaultData(),
            fPlaneNormal( 0.0, 1.0, 0.0 ),
            fPlanePoint( 0.0, 0.0, 0.0 ),
            fSwapAxis( false ),
            fPlaneVectorA( 0.0, 0.0, 1.0 ),
            fPlaneVectorB( 1.0, 0.0, 0.0 ),
            fEpsilon( 1.0e-10 ),
            fROOTSpaces(),
            fROOTSurfaces(),
            fCurrentSpace( NULL ),
            fCurrentSurface( NULL ),
            fCurrentData( NULL ),
            fCurrentOrigin( KThreeVector::sZero ),
            fCurrentXAxis( KThreeVector::sXUnit ),
            fCurrentYAxis( KThreeVector::sYUnit ),
            fCurrentZAxis( KThreeVector::sZUnit ),
            fIgnore( true )
    {
    }
	KGROOTGeometryPainter::~KGROOTGeometryPainter()
    {
        for (size_t i=0; i<fROOTSpaces.size(); i++)
        {
            delete fROOTSpaces.at(i);
        }
        for (size_t i=0; i<fROOTSurfaces.size(); i++)
        {
            delete fROOTSurfaces.at(i);
        }
    }

    void KGROOTGeometryPainter::Render()
    {
    	CalculatePlaneCoordinateSystem();

        KGSurface* tSurface;
        vector< KGSurface* >::iterator tSurfaceIt;
        for( tSurfaceIt = fSurfaces.begin(); tSurfaceIt != fSurfaces.end(); tSurfaceIt++ )
        {
            tSurface = *tSurfaceIt;
            tSurface->AcceptNode( this );
        }

        KGSpace* tSpace;
        vector< KGSpace* >::iterator tSpaceIt;
        for( tSpaceIt = fSpaces.begin(); tSpaceIt != fSpaces.end(); tSpaceIt++ )
        {
            tSpace = *tSpaceIt;
            tSpace->AcceptNode( this );
        }

        return;
    }

    double KGROOTGeometryPainter::GetXMin()
    {
        double tMin(std::numeric_limits<double>::max());
        for (size_t i=0; i<fROOTSpaces.size(); i++)
        {
            TPolyLine* tPolyLine = fROOTSpaces.at( i );
            double* tValue = tPolyLine->GetX();

            for ( int j = 0; j < tPolyLine->Size(); j++ )
            {
                if ( tValue[j] < tMin )
                {
                    tMin = tValue[j];
                }
            }
        }
        for (size_t i=0; i<fROOTSurfaces.size(); i++)
        {
            TPolyLine* tPolyLine = fROOTSurfaces.at( i );
            double* tValue = tPolyLine->GetX();

            for ( int j = 0; j < tPolyLine->Size(); j++ )
            {
                if ( tValue[j] < tMin )
                {
                    tMin = tValue[j];
                }
            }
        }
        return tMin;
    }
    double KGROOTGeometryPainter::GetXMax()
    {
        double tMax(-1.0*std::numeric_limits<double>::max());
        for (size_t i=0; i<fROOTSpaces.size(); i++)
        {
            TPolyLine* tPolyLine = fROOTSpaces.at( i );
            double* tValue = tPolyLine->GetX();

            for ( int j = 0; j < tPolyLine->Size(); j++ )
            {
                if ( tValue[j] > tMax )
                {
                	tMax = tValue[j];
                }
            }
        }
        for (size_t i=0; i<fROOTSurfaces.size(); i++)
        {
            TPolyLine* tPolyLine = fROOTSurfaces.at( i );
            double* tValue = tPolyLine->GetX();

            for ( int j = 0; j < tPolyLine->Size(); j++ )
            {
                if ( tValue[j] > tMax )
                {
                	tMax = tValue[j];
                }
            }
        }
        return tMax;
    }

    double KGROOTGeometryPainter::GetYMin()
    {
        double tMin(std::numeric_limits<double>::max());
        for (size_t i=0; i<fROOTSpaces.size(); i++)
        {
            TPolyLine* tPolyLine = fROOTSpaces.at( i );
            double* tValue = tPolyLine->GetY();

            for ( int j = 0; j < tPolyLine->Size(); j++ )
            {
                if ( tValue[j] < tMin )
                {
                    tMin = tValue[j];
                }
            }
        }
        for (size_t i=0; i<fROOTSurfaces.size(); i++)
        {
            TPolyLine* tPolyLine = fROOTSurfaces.at( i );
            double* tValue = tPolyLine->GetY();

            for ( int j = 0; j < tPolyLine->Size(); j++ )
            {
                if ( tValue[j] < tMin )
                {
                    tMin = tValue[j];
                }
            }
        }
        return tMin;
    }
    double KGROOTGeometryPainter::GetYMax()
    {
        double tMax(-1.0*std::numeric_limits<double>::max());
        for (size_t i=0; i<fROOTSpaces.size(); i++)
        {
            TPolyLine* tPolyLine = fROOTSpaces.at( i );
            double* tValue = tPolyLine->GetY();

            for ( int j = 0; j < tPolyLine->Size(); j++ )
            {
                if ( tValue[j] > tMax )
                {
                	tMax = tValue[j];
                }
            }
        }
        for (size_t i=0; i<fROOTSurfaces.size(); i++)
        {
            TPolyLine* tPolyLine = fROOTSurfaces.at( i );
            double* tValue = tPolyLine->GetY();

            for ( int j = 0; j < tPolyLine->Size(); j++ )
            {
                if ( tValue[j] > tMax )
                {
                	tMax = tValue[j];
                }
            }
        }
        return tMax;
    }


    std::string KGROOTGeometryPainter::GetXAxisLabel()
    {
    	return GetAxisLabel( fPlaneVectorA );
    }
    std::string KGROOTGeometryPainter::GetYAxisLabel()
    {
    	return GetAxisLabel( fPlaneVectorB );
    }

    std::string KGROOTGeometryPainter::GetAxisLabel( KThreeVector anAxis )
    {
    	if ( anAxis.Y() < fEpsilon
    			&& anAxis.Y() > -fEpsilon
    			&& anAxis.Z() < fEpsilon
    			&& anAxis.Z() > -fEpsilon )
    	{
    		if ( anAxis.X() < 1.0 + fEpsilon
    			&& anAxis.X() > 1.0 - fEpsilon)
    		{
				return string( "x" );
    		}
    		if ( anAxis.X() < -1.0 + fEpsilon
    			&& anAxis.X() > -1.0 - fEpsilon)
    		{
				return string( "-x" );
    		}
    	}

    	if ( anAxis.X() < fEpsilon
    			&& anAxis.X() > -fEpsilon
    			&& anAxis.Z() < fEpsilon
    			&& anAxis.Z() > -fEpsilon )
    	{
    		if ( anAxis.Y() < 1.0 + fEpsilon
    			&& anAxis.Y() > 1.0 - fEpsilon)
    		{
				return string( "y" );
    		}
    		if ( anAxis.Y() < -1.0 + fEpsilon
    			&& anAxis.Y() > -1.0 - fEpsilon)
    		{
				return string( "-y" );
    		}
    	}

    	if ( anAxis.X() < fEpsilon
    			&& anAxis.X() > -fEpsilon
    			&& anAxis.Y() < fEpsilon
    			&& anAxis.Y() > -fEpsilon )
    	{
    		if ( anAxis.Z() < 1.0 + fEpsilon
    			&& anAxis.Z() > 1.0 - fEpsilon)
    		{
				return string( "z" );
    		}
    		if ( anAxis.Z() < -1.0 + fEpsilon
    			&& anAxis.Z() > -1.0 - fEpsilon)
    		{
				return string( "-z" );
    		}
    	}

    	string tLabel;
    	stringstream ss;
    	ss << anAxis.X();
    	tLabel += ss.str();
    	tLabel += string( "/" );
    	ss.str("");
    	ss << anAxis.Y();
    	tLabel += ss.str();
    	tLabel += string( "/" );
    	ss.str("");
		ss << anAxis.Z();
		tLabel += ss.str();
    	return tLabel;
    }



    void KGROOTGeometryPainter::Display()
    {

        vismsg( eNormal ) <<"Drawing "<<fROOTSpaces.size()<<" spaces"<<eom;
        for (size_t i=0; i<fROOTSpaces.size(); i++)
        {
        	fROOTSpaces.at(i)->SetFillColor(kGreen+2);
        	fROOTSpaces.at(i)->Draw( "f" );
        }

        vismsg( eNormal ) <<"Drawing "<<fROOTSurfaces.size()<<" surfaces"<<eom;
        for (size_t i=0; i<fROOTSurfaces.size(); i++)
        {
            fROOTSurfaces.at(i)->SetLineColor(kBlack);
            fROOTSurfaces.at(i)->SetLineWidth(1);
            fROOTSurfaces.at(i)->Draw( );
        }
        return;
    }
    void KGROOTGeometryPainter::Write()
    {
    	//root write
        return;
    }

    void KGROOTGeometryPainter::AddSurface( KGSurface* aSurface )
    {
        fSurfaces.push_back( aSurface );
        return;
    }
    void KGROOTGeometryPainter::AddSpace( KGSpace* aSpace )
    {
        fSpaces.push_back( aSpace );
        return;
    }

    void KGROOTGeometryPainter::CalculatePlaneCoordinateSystem()
    {
    	fPlaneNormal = fPlaneNormal.Unit();
    	double tDirectionMagX = fabs( fPlaneNormal.X() );
    	double tDirectionMagY = fabs( fPlaneNormal.Y() );
    	double tDirectionMagZ = fabs( fPlaneNormal.Z() );

    	//plane normal looks in x direction
    	if ( tDirectionMagX >= tDirectionMagY && tDirectionMagX >= tDirectionMagZ )
    	{
			fPlaneVectorA.SetX( 0.0 );
			fPlaneVectorA.SetY( 1.0 );
    		fPlaneVectorA.SetZ( 0.0 );

    		if ( fPlaneNormal.X() > fEpsilon || fPlaneNormal.X() < -1.*fEpsilon )
    		{
    			fPlaneVectorA.SetX( -1.0 * fPlaneNormal.Y() / fPlaneNormal.X() );
    		}

    		fPlaneVectorB.SetX( fPlaneNormal.Y() * fPlaneVectorA.Z() - fPlaneNormal.Z() * fPlaneVectorA.Y() );
    		fPlaneVectorB.SetY( fPlaneNormal.Z() * fPlaneVectorA.X() - fPlaneNormal.X() * fPlaneVectorA.Z() );
    		fPlaneVectorB.SetZ( fPlaneNormal.X() * fPlaneVectorA.Y() - fPlaneNormal.Y() * fPlaneVectorA.X() );

    		fPlaneVectorA = fPlaneVectorA.Unit();
    		fPlaneVectorB = fPlaneVectorB.Unit();

    		if ( fSwapAxis)
    		{
    			swap( fPlaneVectorA, fPlaneVectorB);
    		}
    		vismsg( eNormal ) << "Plane vectors are: "<<fPlaneVectorA<<" and "<<fPlaneVectorB<<eom;
    		if ( fPlaneVectorA.Dot( fPlaneNormal ) > fEpsilon || fPlaneVectorA.Dot( fPlaneNormal ) < -1.*fEpsilon  )
    		{
    			vismsg( eWarning ) <<"Scalar product of PlaneVector A and NormalVector is "<<fPlaneVectorA.Dot( fPlaneNormal )<<eom;
    		}
    		if ( fPlaneVectorB.Dot( fPlaneNormal ) > fEpsilon || fPlaneVectorB.Dot( fPlaneNormal ) < -1.*fEpsilon )
    		{
    			vismsg( eWarning ) <<"Scalar product of PlaneVector B and NormalVector is "<<fPlaneVectorA.Dot( fPlaneNormal )<<eom;
    		}
    		return;
    	}

    	//plane normal looks in y direction
    	if ( tDirectionMagY >= tDirectionMagX && tDirectionMagY >= tDirectionMagZ )
    	{
			fPlaneVectorA.SetX( 0.0 );
			fPlaneVectorA.SetY( 0.0 );
    		fPlaneVectorA.SetZ( 1.0 );

    		if ( fPlaneNormal.Y() > fEpsilon || fPlaneNormal.Y() < -1.*fEpsilon )
    		{
    			fPlaneVectorA.SetY( -1.0 * fPlaneNormal.Z() / fPlaneNormal.Y() );
    		}

    		fPlaneVectorB.SetX( fPlaneNormal.Y() * fPlaneVectorA.Z() - fPlaneNormal.Z() * fPlaneVectorA.Y() );
    		fPlaneVectorB.SetY( fPlaneNormal.Z() * fPlaneVectorA.X() - fPlaneNormal.X() * fPlaneVectorA.Z() );
    		fPlaneVectorB.SetZ( fPlaneNormal.X() * fPlaneVectorA.Y() - fPlaneNormal.Y() * fPlaneVectorA.X() );

    		fPlaneVectorA = fPlaneVectorA.Unit();
    		fPlaneVectorB = fPlaneVectorB.Unit();

    		if ( fSwapAxis)
    		{
    			swap( fPlaneVectorA, fPlaneVectorB);
    		}
    		vismsg( eNormal ) << "Plane vectors are: "<<fPlaneVectorA<<" and "<<fPlaneVectorB<<eom;
    		if ( fPlaneVectorA.Dot( fPlaneNormal ) > fEpsilon || fPlaneVectorA.Dot( fPlaneNormal ) < -1.*fEpsilon  )
    		{
    			vismsg( eWarning ) <<"Scalar product of PlaneVector A and NormalVector is "<<fPlaneVectorA.Dot( fPlaneNormal )<<eom;
    		}
    		if ( fPlaneVectorB.Dot( fPlaneNormal ) > fEpsilon || fPlaneVectorB.Dot( fPlaneNormal ) < -1.*fEpsilon )
    		{
    			vismsg( eWarning ) <<"Scalar product of PlaneVector B and NormalVector is "<<fPlaneVectorA.Dot( fPlaneNormal )<<eom;
    		}
    		return;
    	}

    	//plane normal looks in z direction
    	if ( tDirectionMagZ >= tDirectionMagX && tDirectionMagZ >= tDirectionMagY )
    	{
			fPlaneVectorA.SetX( 1.0 );
			fPlaneVectorA.SetY( 0.0 );
    		fPlaneVectorA.SetZ( 0.0 );

    		if ( fPlaneNormal.Z() > fEpsilon || fPlaneNormal.Z() < -1.*fEpsilon )
    		{
    			fPlaneVectorA.SetZ( -1.0 * fPlaneNormal.X() / fPlaneNormal.Z() );
    		}

    		fPlaneVectorB.SetX( fPlaneNormal.Y() * fPlaneVectorA.Z() - fPlaneNormal.Z() * fPlaneVectorA.Y() );
    		fPlaneVectorB.SetY( fPlaneNormal.Z() * fPlaneVectorA.X() - fPlaneNormal.X() * fPlaneVectorA.Z() );
    		fPlaneVectorB.SetZ( fPlaneNormal.X() * fPlaneVectorA.Y() - fPlaneNormal.Y() * fPlaneVectorA.X() );

    		fPlaneVectorA = fPlaneVectorA.Unit();
    		fPlaneVectorB = fPlaneVectorB.Unit();

    		if ( fSwapAxis)
    		{
    			swap( fPlaneVectorA, fPlaneVectorB);
    		}
    		vismsg( eNormal ) << "Plane vectors are: "<<fPlaneVectorA<<" and "<<fPlaneVectorB<<eom;
    		if ( fPlaneVectorA.Dot( fPlaneNormal ) > fEpsilon || fPlaneVectorA.Dot( fPlaneNormal ) < -1.*fEpsilon  )
    		{
    			vismsg( eWarning ) <<"Scalar product of PlaneVector A and NormalVector is "<<fPlaneVectorA.Dot( fPlaneNormal )<<eom;
    		}
    		if ( fPlaneVectorB.Dot( fPlaneNormal ) > fEpsilon || fPlaneVectorB.Dot( fPlaneNormal ) < -1.*fEpsilon )
    		{
    			vismsg( eWarning ) <<"Scalar product of PlaneVector B and NormalVector is "<<fPlaneVectorA.Dot( fPlaneNormal )<<eom;
    		}
    		return;
    	}
    }

    //****************
    //surface visitors
    //****************

    void KGROOTGeometryPainter::VisitSurface( KGSurface* aSurface )
    {
        fCurrentSurface = aSurface;
        fCurrentOrigin = aSurface->GetOrigin();
        fCurrentXAxis = aSurface->GetXAxis();
        fCurrentYAxis = aSurface->GetYAxis();
        fCurrentZAxis = aSurface->GetZAxis();

        if( aSurface->HasExtension< KGAppearance >() == true )
        {
            fCurrentData = aSurface->AsExtension< KGAppearance >();
        }
        else
        {
            fCurrentData = &fDefaultData;
        }

        if( fCurrentSpace != NULL )
        {
            for( vector< KGSurface* >::const_iterator tIt = fCurrentSpace->GetBoundaries()->begin(); tIt != fCurrentSpace->GetBoundaries()->end(); tIt++ )
            {
                if( (*tIt) == fCurrentSurface )
                {
                    if( fCurrentData == &fDefaultData )
                    {
                        fIgnore = true;
                    }
                    else
                    {
                        fIgnore = false;
                    }
                }
            }
        }
        else
        {
            fIgnore = false;
        }

        return;
    }
    void KGROOTGeometryPainter::VisitFlattenedClosedPathSurface( KGFlattenedCircleSurface* aFlattenedCircleSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create circle points
        ClosedPoints tCirclePoints;
        CircleToClosedPoints( aFlattenedCircleSurface->Path().operator ->(), tCirclePoints );

        //create flattened points
        KThreeVector tApexPoint;
        TubeMesh tMeshPoints;
        ClosedPointsFlattenedToTubeMeshAndApex( tCirclePoints, aFlattenedCircleSurface->Path()->Centroid(), aFlattenedCircleSurface->Z(), tMeshPoints, tApexPoint );

        //create circle and parallel lines
        CircleLines tCircleLines;
        TubeMeshToCircleLines( tMeshPoints, tCircleLines );

        //create intersection from lines
        IntersectionPoints tCircleIntersections;
        LinesToIntersections( tCircleLines, tCircleIntersections );

        OrderedPoints tOrderedPoints;
        IntersectionPointsToOrderedPoints( tCircleIntersections, tOrderedPoints);

        CombineOrderedPoints( tOrderedPoints );

        OrderedPointsToROOTSurface( tOrderedPoints );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGROOTGeometryPainter::VisitFlattenedClosedPathSurface( KGFlattenedPolyLoopSurface* )
    {
		vismsg( eWarning ) << "flattenend polyloop surfaces are not yet supported by the root geometry painter!"<<eom;
        return;
    }
        void KGROOTGeometryPainter::VisitShellPathSurface( KGShellLineSegmentSurface* )
    {
        vismsg( eWarning ) << "shell surfaces are not yet supported by the root geometry painter!"<<eom;
        return;
    }
    void KGROOTGeometryPainter::VisitShellPathSurface( KGShellArcSegmentSurface* )
    {
        vismsg( eWarning ) << "shell surfaces are not yet supported by the root geometry painter!"<<eom;
        return;
    }
    void KGROOTGeometryPainter::VisitShellPathSurface( KGShellPolyLineSurface* )
    {
        vismsg( eWarning ) << "shell surfaces are not yet supported by the root geometry painter!"<<eom;
        return;
    }
    void KGROOTGeometryPainter::VisitShellPathSurface( KGShellPolyLoopSurface* )
    {
        vismsg( eWarning ) << "shell surfaces are not yet supported by the root geometry painter!"<<eom;
        return;
    }
    void KGROOTGeometryPainter::VisitShellPathSurface( KGShellCircleSurface* )
    {
        vismsg( eWarning ) << "shell surfaces are not yet supported by the root geometry painter!"<<eom;
        return;
    }
    void KGROOTGeometryPainter::VisitRotatedPathSurface( KGRotatedLineSegmentSurface* aRotatedLineSegmentSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create line segment points
        OpenPoints tLineSegmentPoints;
        LineSegmentToOpenPoints( aRotatedLineSegmentSurface->Path().operator ->(), tLineSegmentPoints );

        //create rotated points
        TubeMesh tMeshPoints;
        OpenPointsRotatedToTubeMesh( tLineSegmentPoints, tMeshPoints );

        //create circle and parallel lines
        CircleLines tCircleLines;
        ParallelLines tParallelLines;
        TubeMeshToCircleLines( tMeshPoints, tCircleLines );
        TubeMeshToParallelLines( tMeshPoints, tParallelLines );

        //create intersection from lines
        IntersectionPoints tCircleIntersections;
        IntersectionPoints tParallelIntersections;
        LinesToIntersections( tCircleLines, tCircleIntersections );
        LinesToIntersections( tParallelLines, tParallelIntersections );

    	//combine circle and parallel intersections alternating, starting and ending with circle
    	IntersectionPoints tAllIntersections;
    	IntersectionPoints::SetCIt tCircleSetIt = tCircleIntersections.fData.begin();
    	IntersectionPoints::SetCIt tParallelSetIt = tParallelIntersections.fData.begin();
		tAllIntersections.fData.push_back( *tCircleSetIt );
		tCircleSetIt++;
    	while ( tCircleSetIt != tCircleIntersections.fData.end() )
    	{
    		tAllIntersections.fData.push_back( *tParallelSetIt );
    		tAllIntersections.fData.push_back( *tCircleSetIt );
    		tParallelSetIt++;
    		tCircleSetIt++;
    	}

        OrderedPoints tOrderedPoints;
        IntersectionPointsToOrderedPoints( tAllIntersections, tOrderedPoints);

        OrderedPointsToROOTSurface( tOrderedPoints );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGROOTGeometryPainter::VisitRotatedPathSurface( KGRotatedArcSegmentSurface* aRotatedArcSegmentSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create arc segment points
        OpenPoints tArcSegmentPoints;
        ArcSegmentToOpenPoints( aRotatedArcSegmentSurface->Path().operator ->(), tArcSegmentPoints );

        //create rotated points
        TubeMesh tMeshPoints;
        OpenPointsRotatedToTubeMesh( tArcSegmentPoints, tMeshPoints );

        //create circle and parallel lines
        CircleLines tCircleLines;
        ParallelLines tParallelLines;
        TubeMeshToCircleLines( tMeshPoints, tCircleLines );
        TubeMeshToParallelLines( tMeshPoints, tParallelLines );

        //create intersection from lines
        IntersectionPoints tCircleIntersections;
        IntersectionPoints tParallelIntersections;
        LinesToIntersections( tCircleLines, tCircleIntersections );
        LinesToIntersections( tParallelLines, tParallelIntersections );

    	//combine circle and parallel intersections alternating, starting and ending with circle
    	IntersectionPoints tAllIntersections;
    	IntersectionPoints::SetCIt tCircleSetIt = tCircleIntersections.fData.begin();
    	IntersectionPoints::SetCIt tParallelSetIt = tParallelIntersections.fData.begin();
		tAllIntersections.fData.push_back( *tCircleSetIt );
		tCircleSetIt++;
    	while ( tCircleSetIt != tCircleIntersections.fData.end() )
    	{
    		tAllIntersections.fData.push_back( *tParallelSetIt );
    		tAllIntersections.fData.push_back( *tCircleSetIt );
    		tParallelSetIt++;
    		tCircleSetIt++;
    	}

        OrderedPoints tOrderedPoints;
        IntersectionPointsToOrderedPoints( tAllIntersections, tOrderedPoints);

        OrderedPointsToROOTSurface( tOrderedPoints );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGROOTGeometryPainter::VisitRotatedPathSurface( KGRotatedPolyLineSurface* aRotatedPolyLineSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create poly line points
        OpenPoints tPolyLinePoints;
        PolyLineToOpenPoints( aRotatedPolyLineSurface->Path().operator ->(), tPolyLinePoints );

        //create rotated points
        TubeMesh tMeshPoints;
        OpenPointsRotatedToTubeMesh( tPolyLinePoints, tMeshPoints );

        //create circle and parallel lines
        CircleLines tCircleLines;
        ParallelLines tParallelLines;
        TubeMeshToCircleLines( tMeshPoints, tCircleLines );
        TubeMeshToParallelLines( tMeshPoints, tParallelLines );

        //create intersection from lines
        IntersectionPoints tCircleIntersections;
        IntersectionPoints tParallelIntersections;
        LinesToIntersections( tCircleLines, tCircleIntersections );
        LinesToIntersections( tParallelLines, tParallelIntersections );

    	//combine circle and parallel intersections alternating, starting and ending with circle
    	IntersectionPoints tAllIntersections;
    	IntersectionPoints::SetCIt tCircleSetIt = tCircleIntersections.fData.begin();
    	IntersectionPoints::SetCIt tParallelSetIt = tParallelIntersections.fData.begin();
		tAllIntersections.fData.push_back( *tCircleSetIt );
		tCircleSetIt++;
    	while ( tCircleSetIt != tCircleIntersections.fData.end() )
    	{
    		tAllIntersections.fData.push_back( *tParallelSetIt );
    		tAllIntersections.fData.push_back( *tCircleSetIt );
    		tParallelSetIt++;
    		tCircleSetIt++;
    	}

        OrderedPoints tOrderedPoints;
        IntersectionPointsToOrderedPoints( tAllIntersections, tOrderedPoints);

        OrderedPointsToROOTSurface( tOrderedPoints );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGROOTGeometryPainter::VisitRotatedPathSurface( KGRotatedCircleSurface* aRotatedCircleSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create poly line points
        ClosedPoints tCirclePoints;
        CircleToClosedPoints( aRotatedCircleSurface->Path().operator ->(), tCirclePoints );

        //create rotated points
        TorusMesh tMeshPoints;
        ClosedPointsRotatedToTorusMesh( tCirclePoints, tMeshPoints );

        //create circle and parallel lines
        CircleLines tCircleLines;
        ParallelLines tParallelLines;
        TorusMeshToCircleLines( tMeshPoints, tCircleLines );
        TorusMeshToParallelLines( tMeshPoints, tParallelLines );

        //create intersection from lines
        IntersectionPoints tCircleIntersections;
        IntersectionPoints tParallelIntersections;
        LinesToIntersections( tCircleLines, tCircleIntersections );
        LinesToIntersections( tParallelLines, tParallelIntersections );

    	//combine circle and parallel intersections alternating, starting and ending with circle
    	IntersectionPoints tAllIntersections;
    	IntersectionPoints::SetCIt tCircleSetIt = tCircleIntersections.fData.begin();
    	IntersectionPoints::SetCIt tParallelSetIt = tParallelIntersections.fData.begin();
		tAllIntersections.fData.push_back( *tCircleSetIt );
		tCircleSetIt++;
    	while ( tCircleSetIt != tCircleIntersections.fData.end() )
    	{
    		tAllIntersections.fData.push_back( *tParallelSetIt );
    		tAllIntersections.fData.push_back( *tCircleSetIt );
    		tParallelSetIt++;
    		tCircleSetIt++;
    	}

        OrderedPoints tOrderedPoints;
        IntersectionPointsToOrderedPoints( tAllIntersections, tOrderedPoints);

        OrderedPointsToROOTSurface( tOrderedPoints );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGROOTGeometryPainter::VisitRotatedPathSurface( KGRotatedPolyLoopSurface* aRotatedPolyLoopSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create poly loop points
        ClosedPoints tPolyLoopPoints;
        PolyLoopToClosedPoints( aRotatedPolyLoopSurface->Path().operator ->(), tPolyLoopPoints );

        //create rotated points
        TorusMesh tMeshPoints;
        ClosedPointsRotatedToTorusMesh( tPolyLoopPoints, tMeshPoints );

        //create circle and parallel lines
        CircleLines tCircleLines;
        ParallelLines tParallelLines;
        TorusMeshToCircleLines( tMeshPoints, tCircleLines );
        TorusMeshToParallelLines( tMeshPoints, tParallelLines );

        //create intersection from lines
        IntersectionPoints tCircleIntersections;
        IntersectionPoints tParallelIntersections;
        LinesToIntersections( tCircleLines, tCircleIntersections );
        LinesToIntersections( tParallelLines, tParallelIntersections );

    	//combine circle and parallel intersections alternating, starting and ending with circle
    	IntersectionPoints tAllIntersections;
    	IntersectionPoints::SetCIt tCircleSetIt = tCircleIntersections.fData.begin();
    	IntersectionPoints::SetCIt tParallelSetIt = tParallelIntersections.fData.begin();
		tAllIntersections.fData.push_back( *tCircleSetIt );
		tCircleSetIt++;
    	while ( tCircleSetIt != tCircleIntersections.fData.end() )
    	{
    		tAllIntersections.fData.push_back( *tParallelSetIt );
    		tAllIntersections.fData.push_back( *tCircleSetIt );
    		tParallelSetIt++;
    		tCircleSetIt++;
    	}

        OrderedPoints tOrderedPoints;
        IntersectionPointsToOrderedPoints( tAllIntersections, tOrderedPoints);

        //connect last and first point (Todo::This doesnt work out for some cases)
    	for ( OrderedPoints::SetIt tSetIt = tOrderedPoints.fData.begin(); tSetIt != tOrderedPoints.fData.end(); tSetIt++ )
    	{
    		(*tSetIt).fData.push_back( *((*tSetIt).fData.begin()) );
    	}

        OrderedPointsToROOTSurface( tOrderedPoints );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGROOTGeometryPainter::VisitExtrudedPathSurface( KGExtrudedLineSegmentSurface* )
    {
		vismsg( eWarning ) << "extruded surfaces are not yet supported by the root geometry painter!"<<eom;
        return;
    }
    void KGROOTGeometryPainter::VisitExtrudedPathSurface( KGExtrudedArcSegmentSurface* )
    {
		vismsg( eWarning ) << "extruded surfaces are not yet supported by the root geometry painter!"<<eom;
        return;
    }
    void KGROOTGeometryPainter::VisitExtrudedPathSurface( KGExtrudedPolyLineSurface* )
    {
		vismsg( eWarning ) << "extruded surfaces are not yet supported by the root geometry painter!"<<eom;
        return;
    }
    void KGROOTGeometryPainter::VisitExtrudedPathSurface( KGExtrudedCircleSurface* )
    {
		vismsg( eWarning ) << "extruded surfaces are not yet supported by the root geometry painter!"<<eom;
        return;
    }
    void KGROOTGeometryPainter::VisitExtrudedPathSurface( KGExtrudedPolyLoopSurface* )
    {
		vismsg( eWarning ) << "extruded surfaces are not yet supported by the root geometry painter!"<<eom;
        return;
    }

    //**************
    //space visitors
    //**************

    void KGROOTGeometryPainter::VisitSpace( KGSpace* aSpace )
    {
        fCurrentSpace = aSpace;
        fCurrentOrigin = aSpace->GetOrigin();
        fCurrentXAxis = aSpace->GetXAxis();
        fCurrentYAxis = aSpace->GetYAxis();
        fCurrentZAxis = aSpace->GetZAxis();

        if( aSpace->HasExtension< KGAppearance >() == true )
        {
            fCurrentData = aSpace->AsExtension< KGAppearance >();
        }
        else
        {
            fCurrentData = &fDefaultData;
        }

        return;
    }
    void KGROOTGeometryPainter::VisitRotatedOpenPathSpace( KGRotatedLineSegmentSpace* aRotatedLineSegmentSpace )
    {
        //create line segment points
        OpenPoints tLineSegmentPoints;
        LineSegmentToOpenPoints( aRotatedLineSegmentSpace->Path().operator ->(), tLineSegmentPoints );

        //create rotated points
        TubeMesh tMeshPoints;
        OpenPointsRotatedToTubeMesh( tLineSegmentPoints, tMeshPoints );

        //create circle and parallel lines
        CircleLines tCircleLines;
        ParallelLines tParallelLines;
        TubeMeshToCircleLines( tMeshPoints, tCircleLines );
        TubeMeshToParallelLines( tMeshPoints, tParallelLines );

        //create intersection from lines
        IntersectionPoints tCircleIntersections;
        IntersectionPoints tParallelIntersections;
        LinesToIntersections( tCircleLines, tCircleIntersections );
        LinesToIntersections( tParallelLines, tParallelIntersections );

    	//combine circle and parallel intersections alternating, starting and ending with circle
    	IntersectionPoints tAllIntersections;
    	IntersectionPoints::SetCIt tCircleSetIt = tCircleIntersections.fData.begin();
    	IntersectionPoints::SetCIt tParallelSetIt = tParallelIntersections.fData.begin();
		tAllIntersections.fData.push_back( *tCircleSetIt );
		tCircleSetIt++;
    	while ( tCircleSetIt != tCircleIntersections.fData.end() )
    	{
    		tAllIntersections.fData.push_back( *tParallelSetIt );
    		tAllIntersections.fData.push_back( *tCircleSetIt );
    		tParallelSetIt++;
    		tCircleSetIt++;
    	}

        OrderedPoints tOrderedPoints;
        IntersectionPointsToOrderedPoints( tAllIntersections, tOrderedPoints);

        CombineOrderedPoints( tOrderedPoints );

        //connect last and first point (Todo::This doesnt work out for some cases)
    	for ( OrderedPoints::SetIt tSetIt = tOrderedPoints.fData.begin(); tSetIt != tOrderedPoints.fData.end(); tSetIt++ )
    	{
    		(*tSetIt).fData.push_back( *((*tSetIt).fData.begin()) );
    	}

        OrderedPointsToROOTSpace( tOrderedPoints );

        //clear surface
        fCurrentSpace = NULL;

        return;
    }
    void KGROOTGeometryPainter::VisitRotatedOpenPathSpace( KGRotatedArcSegmentSpace* aRotatedArcSegmentSpace )
    {

        //create arc segment points
        OpenPoints tArcSegmentPoints;
        ArcSegmentToOpenPoints( aRotatedArcSegmentSpace->Path().operator ->(), tArcSegmentPoints );

        //create rotated points
        TubeMesh tMeshPoints;
        OpenPointsRotatedToTubeMesh( tArcSegmentPoints, tMeshPoints );

        //create circle and parallel lines
        CircleLines tCircleLines;
        ParallelLines tParallelLines;
        TubeMeshToCircleLines( tMeshPoints, tCircleLines );
        TubeMeshToParallelLines( tMeshPoints, tParallelLines );

        //create intersection from lines
        IntersectionPoints tCircleIntersections;
        IntersectionPoints tParallelIntersections;
        LinesToIntersections( tCircleLines, tCircleIntersections );
        LinesToIntersections( tParallelLines, tParallelIntersections );

    	//combine circle and parallel intersections alternating, starting and ending with circle
    	IntersectionPoints tAllIntersections;
    	IntersectionPoints::SetCIt tCircleSetIt = tCircleIntersections.fData.begin();
    	IntersectionPoints::SetCIt tParallelSetIt = tParallelIntersections.fData.begin();
		tAllIntersections.fData.push_back( *tCircleSetIt );
		tCircleSetIt++;
    	while ( tCircleSetIt != tCircleIntersections.fData.end() )
    	{
    		tAllIntersections.fData.push_back( *tParallelSetIt );
    		tAllIntersections.fData.push_back( *tCircleSetIt );
    		tParallelSetIt++;
    		tCircleSetIt++;
    	}

        OrderedPoints tOrderedPoints;
        IntersectionPointsToOrderedPoints( tAllIntersections, tOrderedPoints);

        CombineOrderedPoints( tOrderedPoints );

        //connect last and first point (Todo::This doesnt work out for some cases)
    	for ( OrderedPoints::SetIt tSetIt = tOrderedPoints.fData.begin(); tSetIt != tOrderedPoints.fData.end(); tSetIt++ )
    	{
    		(*tSetIt).fData.push_back( *((*tSetIt).fData.begin()) );
    	}

        OrderedPointsToROOTSpace( tOrderedPoints );

        //clear surface
        fCurrentSpace = NULL;

        return;
    }

    void KGROOTGeometryPainter::VisitRotatedOpenPathSpace( KGRotatedPolyLineSpace* aRotatedPolyLineSpace )
    {
        //create poly line points
        OpenPoints tPolyLinePoints;
        PolyLineToOpenPoints( aRotatedPolyLineSpace->Path().operator ->(), tPolyLinePoints );

        //create rotated points
        TubeMesh tMeshPoints;
        OpenPointsRotatedToTubeMesh( tPolyLinePoints, tMeshPoints );

        //create circle and parallel lines
        CircleLines tCircleLines;
        ParallelLines tParallelLines;
        TubeMeshToCircleLines( tMeshPoints, tCircleLines );
        TubeMeshToParallelLines( tMeshPoints, tParallelLines );

        //create intersection from lines
        IntersectionPoints tCircleIntersections;
        IntersectionPoints tParallelIntersections;
        LinesToIntersections( tCircleLines, tCircleIntersections );
        LinesToIntersections( tParallelLines, tParallelIntersections );

    	//combine circle and parallel intersections alternating, starting and ending with circle
    	IntersectionPoints tAllIntersections;
    	IntersectionPoints::SetCIt tCircleSetIt = tCircleIntersections.fData.begin();
    	IntersectionPoints::SetCIt tParallelSetIt = tParallelIntersections.fData.begin();
		tAllIntersections.fData.push_back( *tCircleSetIt );
		tCircleSetIt++;
    	while ( tCircleSetIt != tCircleIntersections.fData.end() )
    	{
    		tAllIntersections.fData.push_back( *tParallelSetIt );
    		tAllIntersections.fData.push_back( *tCircleSetIt );
    		tParallelSetIt++;
    		tCircleSetIt++;
    	}

        OrderedPoints tOrderedPoints;
        IntersectionPointsToOrderedPoints( tAllIntersections, tOrderedPoints);

        CombineOrderedPoints( tOrderedPoints );

        //connect last and first point (Todo::This doesnt work out for some cases)
    	for ( OrderedPoints::SetIt tSetIt = tOrderedPoints.fData.begin(); tSetIt != tOrderedPoints.fData.end(); tSetIt++ )
    	{
    		(*tSetIt).fData.push_back( *((*tSetIt).fData.begin()) );
    	}

        OrderedPointsToROOTSpace( tOrderedPoints );

        //clear surface
        fCurrentSpace = NULL;

        return;
    }
    void KGROOTGeometryPainter::VisitRotatedClosedPathSpace( KGRotatedCircleSpace* aRotatedCircleSpace )
    {
        //create poly line points
        ClosedPoints tCirclePoints;
        CircleToClosedPoints( aRotatedCircleSpace->Path().operator ->(), tCirclePoints );

        //create rotated points
        TorusMesh tMeshPoints;
        ClosedPointsRotatedToTorusMesh( tCirclePoints, tMeshPoints );

        //create circle and parallel lines
        CircleLines tCircleLines;
        ParallelLines tParallelLines;
        TorusMeshToCircleLines( tMeshPoints, tCircleLines );
        TorusMeshToParallelLines( tMeshPoints, tParallelLines );

        //create intersection from lines
        IntersectionPoints tCircleIntersections;
        IntersectionPoints tParallelIntersections;
        LinesToIntersections( tCircleLines, tCircleIntersections );
        LinesToIntersections( tParallelLines, tParallelIntersections );

    	//combine circle and parallel intersections alternating, starting and ending with circle
    	IntersectionPoints tAllIntersections;
    	IntersectionPoints::SetCIt tCircleSetIt = tCircleIntersections.fData.begin();
    	IntersectionPoints::SetCIt tParallelSetIt = tParallelIntersections.fData.begin();
		tAllIntersections.fData.push_back( *tCircleSetIt );
		tCircleSetIt++;
    	while ( tCircleSetIt != tCircleIntersections.fData.end() )
    	{
    		tAllIntersections.fData.push_back( *tParallelSetIt );
    		tAllIntersections.fData.push_back( *tCircleSetIt );
    		tParallelSetIt++;
    		tCircleSetIt++;
    	}

        OrderedPoints tOrderedPoints;
        IntersectionPointsToOrderedPoints( tAllIntersections, tOrderedPoints);

        OrderedPointsToROOTSpace( tOrderedPoints );

        //clear surface
        fCurrentSpace = NULL;

        return;
    }
    void KGROOTGeometryPainter::VisitRotatedClosedPathSpace( KGRotatedPolyLoopSpace* aRotatedPolyLoopSpace )
    {
        //create poly loop points
        ClosedPoints tPolyLoopPoints;
        PolyLoopToClosedPoints( aRotatedPolyLoopSpace->Path().operator ->(), tPolyLoopPoints );

        //create rotated points
        TorusMesh tMeshPoints;
        ClosedPointsRotatedToTorusMesh( tPolyLoopPoints, tMeshPoints );

        //create circle and parallel lines
        CircleLines tCircleLines;
        ParallelLines tParallelLines;
        TorusMeshToCircleLines( tMeshPoints, tCircleLines );
        TorusMeshToParallelLines( tMeshPoints, tParallelLines );

        //create intersection from lines
        IntersectionPoints tCircleIntersections;
        IntersectionPoints tParallelIntersections;
        LinesToIntersections( tCircleLines, tCircleIntersections );
        LinesToIntersections( tParallelLines, tParallelIntersections );

    	//combine circle and parallel intersections alternating, starting and ending with circle
    	IntersectionPoints tAllIntersections;
    	IntersectionPoints::SetCIt tCircleSetIt = tCircleIntersections.fData.begin();
    	IntersectionPoints::SetCIt tParallelSetIt = tParallelIntersections.fData.begin();
		tAllIntersections.fData.push_back( *tCircleSetIt );
		tCircleSetIt++;
    	while ( tCircleSetIt != tCircleIntersections.fData.end() )
    	{
    		tAllIntersections.fData.push_back( *tParallelSetIt );
    		tAllIntersections.fData.push_back( *tCircleSetIt );
    		tParallelSetIt++;
    		tCircleSetIt++;
    	}

        OrderedPoints tOrderedPoints;
        IntersectionPointsToOrderedPoints( tAllIntersections, tOrderedPoints);

        //connect last and first point (Todo::This doesnt work out for some cases)
    	for ( OrderedPoints::SetIt tSetIt = tOrderedPoints.fData.begin(); tSetIt != tOrderedPoints.fData.end(); tSetIt++ )
    	{
    		(*tSetIt).fData.push_back( *((*tSetIt).fData.begin()) );
    	}

        OrderedPointsToROOTSpace( tOrderedPoints );

        //clear surface
        fCurrentSpace = NULL;

        return;
    }
    void KGROOTGeometryPainter::VisitExtrudedClosedPathSpace( KGExtrudedCircleSpace* )
    {
		vismsg( eWarning ) << "extruded spaces are not yet supported by the root geometry painter!"<<eom;
        return;
    }

    void KGROOTGeometryPainter::VisitExtrudedClosedPathSpace( KGExtrudedPolyLoopSpace* )
    {
		vismsg( eWarning ) << "extruded spaces are not yet supported by the root geometry painter!"<<eom;
        return;
    }

    void KGROOTGeometryPainter::LocalToGlobal( const KThreeVector& aLocal, KThreeVector& aGlobal )
    {
        aGlobal = fCurrentOrigin + aLocal.X() * fCurrentXAxis + aLocal.Y() * fCurrentYAxis + aLocal.Z() * fCurrentZAxis;
//        vismsg_debug( "Converting "<<aLocal<<" to "<<aGlobal<<eom);
        return;
    }

    //****************
    //points functions
    //****************

    void KGROOTGeometryPainter::LineSegmentToOpenPoints( const KGPlanarLineSegment* aLineSegment, OpenPoints& aPoints )
    {
        aPoints.fData.clear();

        aPoints.fData.push_back( aLineSegment->At( 0. ) );
        aPoints.fData.push_back( aLineSegment->At( aLineSegment->Length() ) );

        vismsg_debug( "line segment partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom );

        return;
    }
    void KGROOTGeometryPainter::ArcSegmentToOpenPoints( const KGPlanarArcSegment* anArcSegment, OpenPoints& aPoints )
    {
        aPoints.fData.clear();

        double tArcFraction = anArcSegment->Length() / (2. * KConst::Pi() * anArcSegment->Radius());
        unsigned int tArc = (unsigned int) (ceil( tArcFraction * (double) (fCurrentData->GetArc()) ));

        double tFraction;
        unsigned int tCount;
        for( tCount = 0; tCount <= tArc; tCount++ )
        {
            tFraction = anArcSegment->Length() * ((double) (tCount) / (double) (tArc));
            aPoints.fData.push_back( anArcSegment->At( tFraction ) );
        }

        vismsg_debug( "arc segment partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom );

        return;
    }
    void KGROOTGeometryPainter::PolyLineToOpenPoints( const KGPlanarPolyLine* aPolyLine, OpenPoints& aPoints )
    {
        aPoints.fData.clear();

        const KGPlanarPolyLine::Set& tElements = aPolyLine->Elements();
        KGPlanarPolyLine::CIt tElementIt;
        const KGPlanarOpenPath* tElement;
        const KGPlanarLineSegment* tLineSegmentElement;
        const KGPlanarArcSegment* tArcSegmentElement;

        OpenPoints tSubPoints;
        for( tElementIt = tElements.begin(); tElementIt != tElements.end(); tElementIt++ )
        {
            tElement = *tElementIt;

            tLineSegmentElement = dynamic_cast< const KGPlanarLineSegment* >( tElement );
            if( tLineSegmentElement != NULL )
            {
                LineSegmentToOpenPoints( tLineSegmentElement, tSubPoints );
                aPoints.fData.insert( aPoints.fData.end(), tSubPoints.fData.begin(), --(tSubPoints.fData.end()) );
                continue;
            }

            tArcSegmentElement = dynamic_cast< const KGPlanarArcSegment* >( tElement );
            if( tArcSegmentElement != NULL )
            {
                ArcSegmentToOpenPoints( tArcSegmentElement, tSubPoints );
                aPoints.fData.insert( aPoints.fData.end(), tSubPoints.fData.begin(), --(tSubPoints.fData.end()) );
                continue;
            }
        }

        aPoints.fData.push_back( aPolyLine->End() );

        vismsg_debug( "poly line partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom );

        return;
    }
    void KGROOTGeometryPainter::CircleToClosedPoints( const KGPlanarCircle* aCircle, ClosedPoints& aPoints )
    {
        aPoints.fData.clear();

        unsigned int tArc = fCurrentData->GetArc();

        double tFraction;
        unsigned int tCount;
        for( tCount = 0; tCount < tArc; tCount++ )
        {
            tFraction = aCircle->Length() * ((double) (tCount) / (double) (tArc));
            aPoints.fData.push_back( aCircle->At( tFraction ) );
        }

        vismsg_debug( "circle partitioned into <" << aPoints.fData.size() << "> closed points vertices" << eom );

        return;
    }
    void KGROOTGeometryPainter::PolyLoopToClosedPoints( const KGPlanarPolyLoop* aPolyLoop, ClosedPoints& aPoints )
    {
        aPoints.fData.clear();

        const KGPlanarPolyLoop::Set& tElements = aPolyLoop->Elements();
        KGPlanarPolyLoop::CIt tElementIt;
        const KGPlanarOpenPath* tElement;
        const KGPlanarLineSegment* tLineSegmentElement;
        const KGPlanarArcSegment* tArcSegmentElement;

        OpenPoints tSubPoints;
        for( tElementIt = tElements.begin(); tElementIt != tElements.end(); tElementIt++ )
        {
            tElement = *tElementIt;
            tSubPoints.fData.clear();

            tLineSegmentElement = dynamic_cast< const KGPlanarLineSegment* >( tElement );
            if( tLineSegmentElement != NULL )
            {
                LineSegmentToOpenPoints( tLineSegmentElement, tSubPoints );
                aPoints.fData.insert( aPoints.fData.end(), tSubPoints.fData.begin(), --(tSubPoints.fData.end()) );
                continue;
            }

            tArcSegmentElement = dynamic_cast< const KGPlanarArcSegment* >( tElement );
            if( tArcSegmentElement != NULL )
            {
                ArcSegmentToOpenPoints( tArcSegmentElement, tSubPoints );
                aPoints.fData.insert( aPoints.fData.end(), tSubPoints.fData.begin(), --(tSubPoints.fData.end()) );
                continue;
            }
        }

        vismsg_debug( "poly loop partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom );

        return;
    }

    //**************
    //mesh functions
    //**************

    void KGROOTGeometryPainter::ClosedPointsFlattenedToTubeMeshAndApex( const ClosedPoints& aPoints, const KTwoVector& aCentroid, const double& aZ, TubeMesh& aMesh, KThreeVector& anApex )
    {
        KThreeVector tPoint;
        TubeMesh::Group tGroup;
        for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tPoint.X() = (*tPointsIt).X();
            tPoint.Y() = (*tPointsIt).Y();
            tPoint.Z() = aZ;
            tGroup.push_back( tPoint );
        }
        aMesh.fData.push_back( tGroup );
        anApex.X() = aCentroid.X();
        anApex.Y() = aCentroid.Y();
        anApex.Z() = aZ;

        vismsg_debug( "flattened closed points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> tube mesh vertices" << eom );

        return;
    }
    void KGROOTGeometryPainter::OpenPointsRotatedToTubeMesh( const OpenPoints& aPoints, TubeMesh& aMesh )
    {
        unsigned int tArc = fCurrentData->GetArc();

        double tFraction;
        unsigned int tCount;

        KThreeVector tPoint;
        TubeMesh::Group tGroup;
        for( OpenPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tGroup.clear();

			//get case, when point is on z axis
			if ( (*tPointsIt).Y() == 0.0 )
			{
				tPoint.X() = 0.0;
				tPoint.Y() = 0.0;
				tPoint.Z() = (*tPointsIt).X();;
				tGroup.push_back( tPoint );
				aMesh.fData.push_back( tGroup );
				continue;
			}

            for( tCount = 0; tCount < tArc; tCount++ )
            {
                tFraction = (double) (tCount) / (double) (tArc);

                tPoint.X() = (*tPointsIt).Y() * cos( 2. * KConst::Pi() * tFraction );
                tPoint.Y() = (*tPointsIt).Y() * sin( 2. * KConst::Pi() * tFraction );
                tPoint.Z() = (*tPointsIt).X();
                tGroup.push_back( tPoint );
            }
            aMesh.fData.push_back( tGroup );
        }

        vismsg_debug( "rotated open points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> tube mesh vertices" << eom );

        return;
    }
    void KGROOTGeometryPainter::ClosedPointsRotatedToTorusMesh( const ClosedPoints& aPoints, TorusMesh& aMesh )
    {
        unsigned int tArc = fCurrentData->GetArc();

        double tFraction;
        unsigned int tCount;

        KThreeVector tPoint;
        TubeMesh::Group tGroup;
        for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tGroup.clear();
            for( tCount = 0; tCount < tArc; tCount++ )
            {
                tFraction = (double) (tCount) / (double) (tArc);

                tPoint.X() = (*tPointsIt).Y() * cos( 2. * KConst::Pi() * tFraction );
                tPoint.Y() = (*tPointsIt).Y() * sin( 2. * KConst::Pi() * tFraction );
                tPoint.Z() = (*tPointsIt).X();
                tGroup.push_back( tPoint );
            }
            aMesh.fData.push_back( tGroup );
        }

        vismsg_debug( "rotated closed points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> torus mesh vertices" << eom );

        return;
    }
    void KGROOTGeometryPainter::OpenPointsExtrudedToFlatMesh( const OpenPoints& aPoints, const double& aZMin, const double& aZMax, FlatMesh& aMesh )
    {
        KThreeVector tPoint;
        TubeMesh::Group tGroup;

        tGroup.clear();
        for( OpenPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tPoint.X() = (*tPointsIt).X();
            tPoint.Y() = (*tPointsIt).Y();
            tPoint.Z() = aZMin;
            tGroup.push_back( tPoint );
        }
        aMesh.fData.push_back( tGroup );

        tGroup.clear();
        for( OpenPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tPoint.X() = (*tPointsIt).X();
            tPoint.Y() = (*tPointsIt).Y();
            tPoint.Z() = aZMax;
            tGroup.push_back( tPoint );
        }
        aMesh.fData.push_back( tGroup );

        vismsg_debug( "extruded open points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> flat mesh vertices" << eom );

        return;
    }
    void KGROOTGeometryPainter::ClosedPointsExtrudedToTubeMesh( const ClosedPoints& aPoints, const double& aZMin, const double& aZMax, TubeMesh& aMesh )
    {
        KThreeVector tPoint;
        TubeMesh::Group tGroup;

        tGroup.clear();
        for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tPoint.X() = (*tPointsIt).X();
            tPoint.Y() = (*tPointsIt).Y();
            tPoint.Z() = aZMin;
            tGroup.push_back( tPoint );
        }
        aMesh.fData.push_back( tGroup );

        tGroup.clear();
        for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tPoint.X() = (*tPointsIt).X();
            tPoint.Y() = (*tPointsIt).Y();
            tPoint.Z() = aZMax;
            tGroup.push_back( tPoint );
        }
        aMesh.fData.push_back( tGroup );

        vismsg_debug( "extruded closed points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> tube mesh vertices" << eom );

        return;
    }

    //**************
    //line functions
    //**************

    void KGROOTGeometryPainter::TubeMeshToCircleLines( const TubeMesh aMesh, CircleLines& aCircleLines )
    {
        //object allocation
        KThreeVector tPoint1, tPoint2;
        Lines::Group tCircleLinesGroup;

        //create lines from tube mesh
        for( TubeMesh::SetCIt tSetIt = aMesh.fData.begin(); tSetIt != aMesh.fData.end(); tSetIt++ )
        {
        	tCircleLinesGroup.clear();
        	TubeMesh::GroupCIt tGroupIt = tSetIt->begin();
        	//get case, when point is on z axis (no tube mesh, just single point)
        	if ( tSetIt->size() == 1 )
        	{
        		LocalToGlobal( *tGroupIt, tPoint1 );
                tCircleLinesGroup.push_back( Lines::Line( tPoint1, tPoint1 ) );
                tCircleLinesGroup.push_back( Lines::Line( tPoint1, tPoint1 ) );
                aCircleLines.fData.push_back( tCircleLinesGroup );
                continue;
        	}
        	while ( ( tGroupIt + 1 ) != tSetIt->end() )
        	{
                LocalToGlobal( *tGroupIt, tPoint1 );
                LocalToGlobal( *(tGroupIt + 1), tPoint2 );
                tCircleLinesGroup.push_back( Lines::Line( tPoint1, tPoint2 ) );
        		tGroupIt++;
        	}
        	//last point and first point
        	tCircleLinesGroup.push_back( Lines::Line( tCircleLinesGroup.back().second, tCircleLinesGroup.front().first ) );
        	aCircleLines.fData.push_back( tCircleLinesGroup );
        }

        vismsg_debug( "tube mesh into <" << aCircleLines.fData.size() << "> circle line groups with <" << tCircleLinesGroup.size() <<"> lines each" << eom );

    }

    void KGROOTGeometryPainter::TubeMeshToParallelLines( const TubeMesh aMesh, ParallelLines& aParallelLines )
    {
        //object allocation
        KThreeVector tPoint1, tPoint2;
        Lines::Group tParallelLinesGroup;

        //create lines from tube mesh
    	TubeMesh::SetCIt tSetIt = aMesh.fData.begin();
    	while( ( tSetIt + 1 ) != aMesh.fData.end() )
    	{
    		tParallelLinesGroup.clear();

    		//get case when user is stupid and creates e.g. a cylinder without radius
    		if ( tSetIt->size() == 1 && ( tSetIt + 1)->size() == 1 )
    		{
    			vismsg ( eWarning ) <<"to less points in tube mesh, check your geometry"<<eom;
    		}

        	//get case, when point is on z axis (no tube mesh, just single point)
        	if ( tSetIt->size() == 1 )
        	{
        		KThreeVector tSinglePoint = *(tSetIt->begin());
				LocalToGlobal( tSinglePoint, tPoint1 );
    			for( TubeMesh::GroupCIt tNextGroupIt = ( tSetIt + 1)->begin(); tNextGroupIt != ( tSetIt + 1)->end(); tNextGroupIt ++ )
    			{
                    LocalToGlobal( *tNextGroupIt, tPoint2 );
                    tParallelLinesGroup.push_back( Lines::Line( tPoint1, tPoint2 ) );
    			}
    			aParallelLines.fData.push_back( tParallelLinesGroup );
    			tSetIt++;
                continue;
        	}

    		TubeMesh::GroupCIt tNextGroupIt = ( tSetIt + 1)->begin();
        	//get case, when next point is on z axis (no tube mesh, just single point)
        	if ( ( tSetIt + 1)->size() == 1 )
        	{
        		KThreeVector tSinglePoint = *( (tSetIt + 1)->begin());
				LocalToGlobal( tSinglePoint, tPoint2 );
				for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
				{
	                LocalToGlobal( *tGroupIt, tPoint1 );
	                tParallelLinesGroup.push_back( Lines::Line( tPoint1, tPoint2 ) );
					tNextGroupIt++;
				}
    			aParallelLines.fData.push_back( tParallelLinesGroup );
    			tSetIt++;
                continue;
        	}

        	//normal case if all groups have same size
			for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
			{
                LocalToGlobal( *tGroupIt, tPoint1 );
                LocalToGlobal( *tNextGroupIt, tPoint2 );
                tParallelLinesGroup.push_back( Lines::Line( tPoint1, tPoint2 ) );
				tNextGroupIt++;
			}
			aParallelLines.fData.push_back( tParallelLinesGroup );
			tSetIt++;
    	}

        vismsg_debug( "tube mesh into <" << aParallelLines.fData.size() << "> parallel line groups with <" << tParallelLinesGroup.size() <<"> lines each" << eom );

    }

    void KGROOTGeometryPainter::TorusMeshToCircleLines( const TorusMesh aMesh, CircleLines& aCircleLines )
    {
        //object allocation
        KThreeVector tPoint1, tPoint2;
        Lines::Group tCircleLinesGroup;

        //create lines from tube mesh
        for( TubeMesh::SetCIt tSetIt = aMesh.fData.begin(); tSetIt != aMesh.fData.end(); tSetIt++ )
        {
        	tCircleLinesGroup.clear();
        	TubeMesh::GroupCIt tGroupIt = tSetIt->begin();
        	while ( ( tGroupIt + 1 ) != tSetIt->end() )
        	{
                LocalToGlobal( *tGroupIt, tPoint1 );
                LocalToGlobal( *(tGroupIt + 1), tPoint2 );
                tCircleLinesGroup.push_back( Lines::Line( tPoint1, tPoint2 ) );
        		tGroupIt++;
        	}
        	//last point and first point
        	tCircleLinesGroup.push_back( Lines::Line( tCircleLinesGroup.back().second, tCircleLinesGroup.front().first ) );
        	aCircleLines.fData.push_back( tCircleLinesGroup );
        }

        vismsg_debug( "tube mesh into <" << aCircleLines.fData.size() << "> circle line groups with <" << tCircleLinesGroup.size() <<"> lines each" << eom );

    }

    void KGROOTGeometryPainter::TorusMeshToParallelLines( const TorusMesh aMesh, ParallelLines& aParallelLines )
    {
        //object allocation
        KThreeVector tPoint1, tPoint2;
        Lines::Group tParallelLinesGroup;

        //create lines from tube mesh
    	TubeMesh::SetCIt tSetIt = aMesh.fData.begin();
    	while( ( tSetIt + 1 ) != aMesh.fData.end() )
    	{
    		tParallelLinesGroup.clear();
    		TubeMesh::GroupCIt tNextGroupIt = ( tSetIt + 1)->begin();
			for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
			{
                LocalToGlobal( *tGroupIt, tPoint1 );
                LocalToGlobal( *tNextGroupIt, tPoint2 );
                tParallelLinesGroup.push_back( Lines::Line( tPoint1, tPoint2 ) );
				tNextGroupIt++;
			}
			aParallelLines.fData.push_back( tParallelLinesGroup );
			tSetIt++;
    	}

        vismsg_debug( "tube mesh into <" << aParallelLines.fData.size() << "> parallel line groups with <" << tParallelLinesGroup.size() <<"> lines each" << eom );

    }


    //**********************
    //intersection functions
    //**********************

    void KGROOTGeometryPainter::LinesToIntersections( const CircleLines aCircleLines, IntersectionPoints& anIntersectionPoints )
    {
    	vismsg_debug( "Calculating intersection of <"<<aCircleLines.fData.size()<<"> circle lines" <<eom);
    	KThreeVector tIntersectionPoint;
    	bool tIntersection;
    	IntersectionPoints::Group tIntersectionPointsGroup;

    	for ( Lines::SetCIt tSetIt = aCircleLines.fData.begin(); tSetIt != aCircleLines.fData.end(); tSetIt++ )
    	{
    		vismsg_debug( "Next Circle Line: "<<eom);
    		tIntersectionPointsGroup.clear();
    		Lines::Group tLinesGroup = *tSetIt;
        	for ( Lines::GroupCIt tGroupIt = tLinesGroup.begin(); tGroupIt != tLinesGroup.end(); tGroupIt++ )
        	{
        		CalculatePlaneIntersection( (*tGroupIt).first, (*tGroupIt).second, tIntersectionPoint, tIntersection );
        		if ( tIntersection )
				{
        			vismsg_debug( "intersection found at "<<tIntersectionPoint<<eom);
        			//convert in 2-axis system of plane
        			KTwoVector tPlanePoint;
        			TransformToPlaneSystem( tIntersectionPoint, tPlanePoint);
        			tIntersectionPointsGroup.push_back( tPlanePoint );
				}
        	}
			IntersectionPoints::NamedGroup tNameGroup( tIntersectionPointsGroup, IntersectionPoints::eCircle);
			anIntersectionPoints.fData.push_back( tNameGroup );
    	}
    	return;
    }

    void KGROOTGeometryPainter::LinesToIntersections( const ParallelLines aParallelLines, IntersectionPoints& anIntersectionPoints )
    {
    	vismsg_debug( "Calculating intersection of <"<<aParallelLines.fData.size()<<"> parallel lines" <<eom);
    	KThreeVector tIntersectionPoint;
    	bool tIntersection;
    	IntersectionPoints::Group tIntersectionPointsGroup;

    	for ( Lines::SetCIt tSetIt = aParallelLines.fData.begin(); tSetIt != aParallelLines.fData.end(); tSetIt++ )
    	{
    		vismsg_debug( "Next Parallel Line: "<<eom);
    		tIntersectionPointsGroup.clear();
    		Lines::Group tLinesGroup = *tSetIt;
        	for ( Lines::GroupCIt tGroupIt = tLinesGroup.begin(); tGroupIt != tLinesGroup.end(); tGroupIt++ )
        	{
        		CalculatePlaneIntersection( (*tGroupIt).first, (*tGroupIt).second, tIntersectionPoint, tIntersection );
        		if ( tIntersection )
				{
        			vismsg_debug( "intersection found at "<<tIntersectionPoint<<eom);
        			//convert in 2-axis system of plane
        			KTwoVector tPlanePoint;
        			TransformToPlaneSystem( tIntersectionPoint, tPlanePoint);
        			tIntersectionPointsGroup.push_back( tPlanePoint );
				}
        	}
			IntersectionPoints::NamedGroup tNameGroup( tIntersectionPointsGroup, IntersectionPoints::eParallel);
			anIntersectionPoints.fData.push_back( tNameGroup );
    	}
    	return;
    }


    void KGROOTGeometryPainter::CalculatePlaneIntersection( const KThreeVector aStartPoint, const KThreeVector anEndPoint, KThreeVector& anIntersectionPoint, bool& anIntersection )
    {
    	//calculates the intersection between the line from aStartPoint to anEndPoint with the plane define by tPlaneNormal and tPlanePoint
    	//formula: ( aStartPoint + lambda * ( anEndPoint - aStartPoint ) - fPlanePoint ) * fPlaneNormal = 0
    	//solve for lamda

//    	vismsg_debug( "StartPoint: "<<aStartPoint<<eom);
//    	vismsg_debug( "EndPoint: "<<anEndPoint<<eom);

    	if ( aStartPoint == anEndPoint )
    	{
    		anIntersection = true;
    		anIntersectionPoint = aStartPoint;
    		return;
    	}

    	KThreeVector tLineConnection = anEndPoint - aStartPoint;

    	double tNumerator = fPlaneNormal.X() * ( fPlanePoint.X() - aStartPoint.X() )
						+ fPlaneNormal.Y() * ( fPlanePoint.Y() - aStartPoint.Y() )
						+ fPlaneNormal.Z() * ( fPlanePoint.Z() - aStartPoint.Z() );
    	double tDenominator = tLineConnection.X() * fPlaneNormal.X()
    						+ tLineConnection.Y() * fPlaneNormal.Y()
    						+ tLineConnection.Z() * fPlaneNormal.Z();


    	if ( tDenominator < fEpsilon && tDenominator > -1.0 * fEpsilon )
    	{
    		//plane and line parallel
    		anIntersection = false;
    		return;
    	}
    	else
    	{
    		anIntersection = true;
    	}
    	double tLambda = tNumerator / tDenominator;

//    	vismsg_debug( tNumerator<<"\t"<<tDenominator<<"\t"<<tLambda<<eom );

    	//line is parallel but on plane
//    	if ( tNumerator < fEpsilon && tNumerator > -1.0 * fEpsilon )
//    	{
//    		tLambda = 0.0;
//    	}

    	if ( tLambda > -fEpsilon && tLambda < 1.0 - fEpsilon )
    	{
//    		vismsg_debug( "found intersection, lamda is "<<tLambda<<eom);
    		anIntersection = true;
    	}
    	else
    	{
//    		vismsg_debug( "found no intersection, lamda is "<<tLambda<<eom);
    		anIntersection = false;
    		return;
    	}

    	anIntersectionPoint.SetX( aStartPoint.X() + tLambda * tLineConnection.X() );
    	anIntersectionPoint.SetY( aStartPoint.Y() + tLambda * tLineConnection.Y() );
    	anIntersectionPoint.SetZ( aStartPoint.Z() + tLambda * tLineConnection.Z() );

//    	vismsg_debug( anIntersection<<"\t"<<anIntersectionPoint.X()<<"\t"<<anIntersectionPoint.Y()<<"\t"<<anIntersectionPoint.Z()<<eom);

    	return;
    }

    void KGROOTGeometryPainter::TransformToPlaneSystem( const KThreeVector aPoint, KTwoVector& aPlanePoint )
    {
    	//solve aPoint = fPlanePoint + alpha * fPlaneA + beta * fPlaneB for alpha and beta
    	double tAlpha, tBeta;

    	if ( ( fPlaneVectorA.X() * fPlaneVectorB.Y() - fPlaneVectorA.Y() * fPlaneVectorB.X() ) != 0.0 )
    	{
    		tAlpha = ( ( aPoint.X() - fPlanePoint.X() ) * fPlaneVectorB.Y() - aPoint.Y() * fPlaneVectorB.X() + fPlanePoint.Y()*fPlaneVectorB.X() )
    				/ ( fPlaneVectorA.X() * fPlaneVectorB.Y() - fPlaneVectorA.Y() * fPlaneVectorB.X() );

    		if ( fPlaneVectorB.Y() != 0)
    		{
				tBeta = ( aPoint.Y() - fPlanePoint.Y() - tAlpha * fPlaneVectorA.Y() )
						/ fPlaneVectorB.Y();
    		}
    		else
    		{
				tBeta = ( aPoint.X() - fPlanePoint.X() - tAlpha * fPlaneVectorA.X() )
						/ fPlaneVectorB.X();
    		}

        	aPlanePoint.SetComponents( tAlpha, tBeta );
            vismsg_debug( "Converting "<<aPoint<<" to "<<aPlanePoint<<eom);
            return;
    	}

    	if ( ( fPlaneVectorA.Y() * fPlaneVectorB.Z() - fPlaneVectorA.Z() * fPlaneVectorB.Y() ) != 0.0 )
    	{
    		tAlpha = ( ( aPoint.Y() - fPlanePoint.Y() ) * fPlaneVectorB.Z() - aPoint.Z() * fPlaneVectorB.Y() + fPlanePoint.Z()*fPlaneVectorB.Y() )
    				/ ( fPlaneVectorA.Y() * fPlaneVectorB.Z() - fPlaneVectorA.Z() * fPlaneVectorB.Y() );

    		if ( fPlaneVectorB.Z() != 0)
    		{
				tBeta = ( aPoint.Z() - fPlanePoint.Z() - tAlpha * fPlaneVectorA.Z() )
						/ fPlaneVectorB.Z();
    		}
    		else
    		{
				tBeta = ( aPoint.Y() - fPlanePoint.Y() - tAlpha * fPlaneVectorA.Y() )
						/ fPlaneVectorB.Y();
    		}

        	aPlanePoint.SetComponents( tAlpha, tBeta );
            vismsg_debug( "Converting "<<aPoint<<" to "<<aPlanePoint<<eom);
            return;
    	}

    	if ( ( fPlaneVectorA.X() * fPlaneVectorB.Z() - fPlaneVectorA.Z() * fPlaneVectorB.X() ) != 0.0 )
    	{
    		tAlpha = ( ( aPoint.X() - fPlanePoint.X() ) * fPlaneVectorB.Z() - aPoint.Z() * fPlaneVectorB.X() + fPlanePoint.Z()*fPlaneVectorB.X() )
    				/ ( fPlaneVectorA.X() * fPlaneVectorB.Z() - fPlaneVectorA.Z() * fPlaneVectorB.X() );

    		if ( fPlaneVectorB.Z() != 0)
    		{
				tBeta = ( aPoint.Z() - fPlanePoint.Z() - tAlpha * fPlaneVectorA.Z() )
						/ fPlaneVectorB.Z();
    		}
    		else
    		{
				tBeta = ( aPoint.X() - fPlanePoint.X() - tAlpha * fPlaneVectorA.X() )
						/ fPlaneVectorB.X();
    		}

        	aPlanePoint.SetComponents( tAlpha, tBeta );
            vismsg_debug( "Converting "<<aPoint<<" to "<<aPlanePoint<<eom);
            return;
    	}

    	vismsg( eWarning ) <<"this should never be called - problem in TransformToPlaneSystem function"<<eom;
    	return;
    }

    void KGROOTGeometryPainter::IntersectionPointsToOrderedPoints( const IntersectionPoints anIntersectionPoints, OrderedPoints& anOrderedPoints )
    {
    	//the intersection points now have to be sorted, to create one (or more) ordered points (which later can be connected to form a shape on the canvas)
    	//3 possible cases here
    	//case 1: Closed ordered points ( data starts and ends with parallel intersection points (or just one group of parallel ))
    	//case 2: Open ordered points ( data starts with parallel intersectionpoints and ends with circle intersection points, or vice versa )
    	//case 3: Dual ordered points ( data has only circle intersection points )

        IntersectionPoints::Origin tStartState = IntersectionPoints::eUndefined;
        IntersectionPoints::Origin tEndState = IntersectionPoints::eUndefined;

        IntersectionPoints tNewIntersectionPoints;

        vismsg_debug ("Trying to order the following intersection points: "<<eom);

#ifdef KGeoBag_ENABLE_DEBUG
    	for ( IntersectionPoints::SetCIt tSetIt = anIntersectionPoints.fData.begin(); tSetIt != anIntersectionPoints.fData.end(); tSetIt++)
    	{
    		IntersectionPoints::NamedGroup tNamedGroup = *tSetIt;
    		IntersectionPoints::Origin tOrigin = tNamedGroup.second;
    		IntersectionPoints::Group tGroup = tNamedGroup.first;
    		vismsg_debug("type <"<<tOrigin<<"> with size<"<<tGroup.size()<<">"<<eom);
    	}
#endif

    	vismsg_debug("start now"<<eom);

    	for ( IntersectionPoints::SetCIt tSetIt = anIntersectionPoints.fData.begin(); tSetIt != anIntersectionPoints.fData.end(); tSetIt++)
    	{
    		IntersectionPoints::NamedGroup tNamedGroup = *tSetIt;
    		IntersectionPoints::Origin tOrigin = tNamedGroup.second;
    		IntersectionPoints::Group tGroup = tNamedGroup.first;
    		vismsg_debug("type <"<<tOrigin<<"> with size<"<<tGroup.size()<<">"<<eom);

			if ( tGroup.size() > 1 )
			{
				if ( tStartState == IntersectionPoints::eUndefined )
				{
					tStartState = tOrigin;
					tEndState = tOrigin;
					tNewIntersectionPoints.fData.push_back( tNamedGroup );
				}
				else
				{
					tEndState = tOrigin;
					tNewIntersectionPoints.fData.push_back( tNamedGroup );
				}

			}


			if ( ( tGroup.size() <= 1 && tOrigin == IntersectionPoints::eCircle && tStartState != IntersectionPoints::eUndefined ) || tSetIt == anIntersectionPoints.fData.end() - 1  )
			{
				//which case do we have?
				if ( tStartState == IntersectionPoints::eParallel && tEndState == IntersectionPoints::eParallel )
				{
					//case 1, closed ordered points
#ifdef KGeoBag_ENABLE_DEBUG
			    	for ( IntersectionPoints::SetIt tTestIt = tNewIntersectionPoints.fData.begin(); tTestIt != tNewIntersectionPoints.fData.end(); tTestIt++)
			    	{
			    		IntersectionPoints::NamedGroup tNamedGroup = *tTestIt;
			    		IntersectionPoints::Origin tOrigin = tNamedGroup.second;
			    		vismsg_debug( "group type is: "<<tOrigin<<eom;)
			    	}
#endif
					vismsg_debug ("Creating closed ordered points"<<eom);
					CreateClosedOrderedPoints( tNewIntersectionPoints, anOrderedPoints );
				}
				if ( tStartState == IntersectionPoints::eParallel && tEndState == IntersectionPoints::eCircle )
				{
					//case 2, open ordered points
#ifdef KGeoBag_ENABLE_DEBUG
			    	for ( IntersectionPoints::SetIt tTestIt = tNewIntersectionPoints.fData.begin(); tTestIt != tNewIntersectionPoints.fData.end(); tTestIt++)
			    	{
			    		IntersectionPoints::NamedGroup tNamedGroup = *tTestIt;
			    		IntersectionPoints::Origin tOrigin = tNamedGroup.second;
			    		vismsg_debug( "group type is: "<<tOrigin<<eom;)
			    	}
#endif
					vismsg_debug ("Creating open ordered points"<<eom);
			    	CreateOpenOrderedPoints( tNewIntersectionPoints, anOrderedPoints );
				}
				if ( tStartState == IntersectionPoints::eCircle && tEndState == IntersectionPoints::eParallel )
				{
					//case 2, reversed open ordered points
#ifdef KGeoBag_ENABLE_DEBUG
			    	for ( IntersectionPoints::SetIt tTestIt = tNewIntersectionPoints.fData.begin(); tTestIt != tNewIntersectionPoints.fData.end(); tTestIt++)
			    	{
			    		IntersectionPoints::NamedGroup tNamedGroup = *tTestIt;
			    		IntersectionPoints::Origin tOrigin = tNamedGroup.second;
			    		vismsg_debug( "group type is: "<<tOrigin<<eom;)
			    	}
#endif
					vismsg_debug ("Creating reversed open ordered points"<<eom);
					reverse( tNewIntersectionPoints.fData.begin(), tNewIntersectionPoints.fData.end() );
			    	CreateOpenOrderedPoints( tNewIntersectionPoints, anOrderedPoints );
				}
				if ( tStartState == IntersectionPoints::eCircle && tEndState == IntersectionPoints::eCircle )
				{
					//case 3, dual ordered points
#ifdef KGeoBag_ENABLE_DEBUG
			    	for ( IntersectionPoints::SetIt tTestIt = tNewIntersectionPoints.fData.begin(); tTestIt != tNewIntersectionPoints.fData.end(); tTestIt++)
			    	{
			    		IntersectionPoints::NamedGroup tNamedGroup = *tTestIt;
			    		IntersectionPoints::Origin tOrigin = tNamedGroup.second;
			    		vismsg_debug( "group type is: "<<tOrigin<<eom;)
			    	}
#endif
					vismsg_debug ("Creating dual ordered points"<<eom);
					CreateDualOrderedPoints( tNewIntersectionPoints, anOrderedPoints );
				}

				tStartState = IntersectionPoints::eUndefined;
				tEndState = IntersectionPoints::eUndefined;
				tNewIntersectionPoints.fData.clear();
			}
    	}

    	if ( anOrderedPoints.fData.size() == 0)
    	{
			vismsg_debug ("plane did not cross geometry"<<eom);
    	}

    	return;
    }

    void KGROOTGeometryPainter::CreateClosedOrderedPoints( const IntersectionPoints anIntersectionPoints, OrderedPoints& anOrderedPoints )
    {
    	ClosedPoints tClosedPoints;
    	//first get the case where there is only one parallel intersection group
    	if ( anIntersectionPoints.fData.size() == 1 )
    	{
    		IntersectionPoints::NamedGroup tNamedGroup = *( anIntersectionPoints.fData.begin() );
    		IntersectionPoints::Origin tOrigin = tNamedGroup.second;
    		IntersectionPoints::Group tGroup = tNamedGroup.first;
    		if ( tOrigin != IntersectionPoints::eParallel )
    		{
				vismsg( eWarning ) <<"debug this?!?"<<eom;
    		}
    		tClosedPoints.fData.insert( tClosedPoints.fData.end(), tGroup.begin(), tGroup.end() );
    		//connect first and last point
    		KTwoVector tFirstPoint = *tClosedPoints.fData.begin();
    		tClosedPoints.fData.push_back( tFirstPoint );

        	anOrderedPoints.fData.push_back( tClosedPoints );
        	//done, this was easy
        	return;
    	}


    	//first create some open points with the intersectionpoints without the last parallel intersection
    	IntersectionPoints tTempIntersectionPoints;
    	tTempIntersectionPoints.fData.insert( tTempIntersectionPoints.fData.end(), anIntersectionPoints.fData.begin(), anIntersectionPoints.fData.end() - 1 );
    	CreateOpenOrderedPoints( tTempIntersectionPoints, anOrderedPoints );

    	Points tPoints = *( anOrderedPoints.fData.end() - 1 );
    	tClosedPoints.fData.insert( tClosedPoints.fData.end(), tPoints.fData.begin(), tPoints.fData.end() );
    	anOrderedPoints.fData.pop_back();

    	//now we just have to add the last parallel intersection to the points and connect them
		vismsg_debug( "Inserting last parallel intersection group "<<eom);
		KTwoVector tLastOpenPoint = *(tClosedPoints.fData.end() - 1);
		vismsg_debug( tLastOpenPoint<<eom);
		IntersectionPoints::SetCIt tSetIt = anIntersectionPoints.fData.end() - 1;
		//look for the points in the group that is closest to the first circle points
		IntersectionPoints::GroupCIt tStartGroupIt;
		double tMin = 1e10;
		for ( IntersectionPoints::GroupCIt tGroupIt = (*tSetIt).first.begin(); tGroupIt != (*tSetIt).first.end(); tGroupIt++ )
		{
			KTwoVector tPoint = *tGroupIt;
			double tTempMin = ( tPoint - tLastOpenPoint ).MagnitudeSquared();
			if (  tTempMin < tMin )
			{
				tMin = tTempMin;
				tStartGroupIt = tGroupIt;
			}
		}
		vismsg_debug( "closest point is "<<*tStartGroupIt<<eom);


		//now sort the last parallel group in the right order
		OpenPoints tLastParallelGroup;
		tLastParallelGroup.fData.insert( tLastParallelGroup.fData.end(), tStartGroupIt, (*tSetIt).first.end() );
		tLastParallelGroup.fData.insert( tLastParallelGroup.fData.end(), (*tSetIt).first.begin(), tStartGroupIt );


		if ( tLastParallelGroup.fData.size() >= 2 )
		{
			//check which rotation direction this group has
			KTwoVector tStartPoint = *(tLastParallelGroup.fData.begin());
			KTwoVector tNextPoint = *(tLastParallelGroup.fData.begin() + 1);
			KTwoVector tLastPoint = *(tLastParallelGroup.fData.end() - 1);
			if ( (tStartPoint - tNextPoint).MagnitudeSquared() > (tStartPoint - tLastPoint).MagnitudeSquared() )
			{
				//lets reverte the group (but keep the start point)
				vismsg_debug( "Reverting because: "<<eom);
				vismsg_debug( "StartPoint "<<tStartPoint<<eom);
				vismsg_debug( "NextPoint "<<tNextPoint<<eom);
				vismsg_debug( "LastPoint "<<tLastPoint<<eom);
				reverse( tLastParallelGroup.fData.begin() + 1, tLastParallelGroup.fData.end() );
			}
		}

		//and put that stuff into the main OpenPoints
		tClosedPoints.fData.insert( tClosedPoints.fData.end() ,tLastParallelGroup.fData.begin(), tLastParallelGroup.fData.end() );


//		for( OpenPoints::It tIt = tClosedPoints.fData.begin(); tIt != tClosedPoints.fData.end(); tIt++ )
//		{
//			vismsg( eNormal ) <<*tIt<<eom;
//		}

		//connect first and last point
		KTwoVector tFirstPoint = *tClosedPoints.fData.begin();
		tClosedPoints.fData.push_back( tFirstPoint );

    	anOrderedPoints.fData.push_back( tClosedPoints );

    	return;
    }

    void KGROOTGeometryPainter::CreateOpenOrderedPoints( const IntersectionPoints anIntersectionPoints, OrderedPoints& anOrderedPoints )
    {
    	OpenPoints tOpenPoints;

    	//keep an iterater to the high point of the first circle
    	KTwoVector tHighPointFirstCircle;

		vismsg_debug( "Inserting the circle intersection points "<<eom);
    	for( IntersectionPoints::SetCIt tSetIt = anIntersectionPoints.fData.begin(); tSetIt != anIntersectionPoints.fData.end(); tSetIt++ )
    	{
    		IntersectionPoints::NamedGroup tNamedGroup = *tSetIt;
    		IntersectionPoints::Origin tOrigin = tNamedGroup.second;
    		IntersectionPoints::Group tGroup = tNamedGroup.first;
    		if ( tOrigin == IntersectionPoints::eCircle )
    		{
				if ( tGroup.size() != 2 )
				{
					vismsg( eWarning ) <<"debug this?!?"<<eom;
					continue;
				}
				KTwoVector tPoint1 = tGroup.front();
				KTwoVector tPoint2 = tGroup.back();
				vismsg_debug( tPoint1<<eom);
				vismsg_debug( tPoint2<<eom);
				if ( tPoint1.X() + tPoint1.Y() > tPoint2.X() + tPoint2.Y() )
				{
					tOpenPoints.fData.push_front( tPoint1 );
					tOpenPoints.fData.push_back( tPoint2 );
				}
				else
				{
					tOpenPoints.fData.push_front( tPoint2 );
					tOpenPoints.fData.push_back( tPoint1 );
				}

				if ( tSetIt == anIntersectionPoints.fData.begin() + 1 )
				{
					tHighPointFirstCircle = *(tOpenPoints.fData.begin());
				}
    		}
    	}

//    	for( OpenPoints::It tIt = tOpenPoints.fData.begin(); tIt != tOpenPoints.fData.end(); tIt++ )
//    	{
//    		vismsg( eNormal ) <<*tIt<<eom;
//    	}

    	//now insert the parallel intersection points
    	OpenPoints::It tOpenPointsFrontIt = tOpenPoints.fData.begin();
    	OpenPoints::It tOpenPointsBackIt = tOpenPoints.fData.end() - 1;

		vismsg_debug( "Inserting the parallel intersection points "<<eom);
    	//loop ignores the first group (that should be a parallel group and connect the two poly line halfes )
		for( IntersectionPoints::SetCIt tSetIt = anIntersectionPoints.fData.begin() + 1; tSetIt != anIntersectionPoints.fData.end(); tSetIt++ )
    	{
    		IntersectionPoints::NamedGroup tNamedGroup = *tSetIt;
    		IntersectionPoints::Origin tOrigin = tNamedGroup.second;
    		IntersectionPoints::Group tGroup = tNamedGroup.first;
    		if ( tOrigin == IntersectionPoints::eParallel )
    		{
    			//find those parallel intersection points that sit between the two circle points
				KTwoVector tCirclePointHigh = *tOpenPointsFrontIt;
				KTwoVector tCirclePointLow = *(tOpenPointsFrontIt+1);
	    		vismsg_debug( "upper part:" <<eom);
				vismsg_debug( "High point "<<tCirclePointHigh<<eom);
				vismsg_debug( "Low point "<<tCirclePointLow<<eom);
				//map here, because autosort
				map<double, KTwoVector> tTempPointsMap;
				OpenPoints tTempPoints;

	    		vismsg_debug( "have points:" <<eom);
		    	for( IntersectionPoints::GroupIt tGroupIt = tGroup.begin(); tGroupIt != tGroup.end(); tGroupIt++ )
		    	{
		    		KTwoVector tPoint = *tGroupIt;
		    		vismsg_debug( tPoint <<eom);
					if ( ( ( tPoint.X() < tCirclePointHigh.X() && tPoint.X() > tCirclePointLow.X() )
							|| ( tPoint.X() > tCirclePointHigh.X() && tPoint.X() < tCirclePointLow.X() ) )
							&& ( ( tPoint.Y() < tCirclePointHigh.Y() && tPoint.Y() > tCirclePointLow.Y() )
							|| ( tPoint.Y() > tCirclePointHigh.Y() && tPoint.Y() < tCirclePointLow.Y() ) ) )
					{
						tTempPointsMap.insert( pair<double,KTwoVector>(tPoint.X(), tPoint ) );
					}
		    	}

	    		vismsg_debug( "using:" <<eom);
		    	for ( map<double,KTwoVector>::iterator tMapIt = tTempPointsMap.begin(); tMapIt != tTempPointsMap.end(); tMapIt++ )
		    	{
		    		vismsg_debug( (*tMapIt).second <<eom);
		    		if ( tCirclePointHigh.X() < tCirclePointLow.X() )
		    		{
		    			tTempPoints.fData.push_back( (*tMapIt).second );
		    		}
		    		else
		    		{
		    			tTempPoints.fData.push_front( (*tMapIt).second );
		    		}
		    	}

		    	//now insert in tOpenPoints
		    	tOpenPoints.fData.insert(tOpenPointsFrontIt + 1, tTempPoints.fData.begin(), tTempPoints.fData.end() );

		    	//the same for the other end of the deque
				tCirclePointHigh = *(tOpenPointsBackIt);
				tCirclePointLow = *(tOpenPointsBackIt - 1);
	    		vismsg_debug( "lower part:" <<eom);
				vismsg_debug( "High point "<<tCirclePointHigh<<eom);
				vismsg_debug( "Low point "<<tCirclePointLow<<eom);
				tTempPointsMap.clear();
				tTempPoints.fData.clear();

	    		vismsg_debug( "have points:" <<eom);
		    	for( IntersectionPoints::GroupIt tGroupIt = tGroup.begin(); tGroupIt != tGroup.end(); tGroupIt++ )
		    	{
		    		KTwoVector tPoint = *tGroupIt;
		    		vismsg_debug( tPoint <<eom);
					if ( ( ( tPoint.X() < tCirclePointHigh.X() && tPoint.X() > tCirclePointLow.X() )
							|| ( tPoint.X() > tCirclePointHigh.X() && tPoint.X() < tCirclePointLow.X() ) )
							&& ( ( tPoint.Y() < tCirclePointHigh.Y() && tPoint.Y() > tCirclePointLow.Y() )
							|| ( tPoint.Y() > tCirclePointHigh.Y() && tPoint.Y() < tCirclePointLow.Y() ) ) )
					{
						tTempPointsMap.insert( pair<double,KTwoVector>(tPoint.X(), tPoint ) );
					}
		    	}

	    		vismsg_debug( "using:" <<eom);
		    	for ( map<double,KTwoVector>::iterator tMapIt = tTempPointsMap.begin(); tMapIt != tTempPointsMap.end(); tMapIt++ )
		    	{
		    		vismsg_debug( (*tMapIt).second <<eom);
		    		if ( tCirclePointHigh.X() > tCirclePointLow.X() )
		    		{
		    			tTempPoints.fData.push_back( (*tMapIt).second );
		    		}
		    		else
		    		{
		    			tTempPoints.fData.push_front( (*tMapIt).second );
		    		}
		    	}

		    	//now insert in tOpenPoints
		    	tOpenPoints.fData.insert(tOpenPointsBackIt, tTempPoints.fData.begin(), tTempPoints.fData.end() );

				tOpenPointsFrontIt++;
				tOpenPointsBackIt--;
    		}
    	}

//		for( OpenPoints::It tIt = tOpenPoints.fData.begin(); tIt != tOpenPoints.fData.end(); tIt++ )
//		{
//			vismsg( eNormal ) <<*tIt<<eom;
//		}

		//now we treat the first parallel group
		vismsg_debug( "Inserting first parallel intersection group "<<eom);
		KTwoVector tFirstCirclePoint = tHighPointFirstCircle;
		vismsg_debug( tFirstCirclePoint<<eom);
		IntersectionPoints::SetCIt tSetIt = anIntersectionPoints.fData.begin();
		//look for the points in the group that is closest to the first circle points
		IntersectionPoints::GroupCIt tStartGroupIt;
		double tMin = std::numeric_limits<double>::max();
		for ( IntersectionPoints::GroupCIt tGroupIt = (*tSetIt).first.begin(); tGroupIt != (*tSetIt).first.end(); tGroupIt++ )
		{
			KTwoVector tPoint = *tGroupIt;
			double tTempMin = ( tPoint - tFirstCirclePoint ).MagnitudeSquared();
			if ( tTempMin < tMin )
			{
				tMin = tTempMin;
				tStartGroupIt = tGroupIt;
			}
		}
		vismsg_debug( "closest point is "<<*tStartGroupIt<<eom);


		//now sort the last parallel group in the right order
		OpenPoints tLastParallelGroup;
		tLastParallelGroup.fData.insert( tLastParallelGroup.fData.end(), tStartGroupIt, (*tSetIt).first.end() );
		tLastParallelGroup.fData.insert( tLastParallelGroup.fData.end(), (*tSetIt).first.begin(), tStartGroupIt );


		if ( tLastParallelGroup.fData.size() >= 2 )
		{
			//check which rotation direction this group has
			KTwoVector tStartPoint = *(tLastParallelGroup.fData.begin());
			KTwoVector tNextPoint = *(tLastParallelGroup.fData.begin() + 1);
			KTwoVector tLastPoint = *(tLastParallelGroup.fData.end() - 1);
			if ( (tStartPoint - tNextPoint).MagnitudeSquared() > (tStartPoint - tLastPoint).MagnitudeSquared() )
			{
				//lets reverte the group (but keep the start point)
				vismsg_debug( "Reverting because: "<<eom);
				vismsg_debug( "StartPoint "<<tStartPoint<<eom);
				vismsg_debug( "NextPoint "<<tNextPoint<<eom);
				vismsg_debug( "LastPoint "<<tLastPoint<<eom);
				reverse( tLastParallelGroup.fData.begin() + 1, tLastParallelGroup.fData.end() );
			}
		}

		OpenPoints::It tHighPointFirstCircleIt;
		for( OpenPoints::It tIt = tOpenPoints.fData.begin(); tIt != tOpenPoints.fData.end(); tIt++ )
		{
			KTwoVector tPoint = *tIt;
			if ( fabs( tPoint.X() - tHighPointFirstCircle.X()) < fEpsilon && fabs( tPoint.Y() - tHighPointFirstCircle.Y()) < fEpsilon  )
			{
				tHighPointFirstCircleIt = tIt;
				break;
			}
		}

		//and put that stuff into the main OpenPoints
		tOpenPoints.fData.insert( tHighPointFirstCircleIt + 1, tLastParallelGroup.fData.begin(), tLastParallelGroup.fData.end() );

//		for( OpenPoints::It tIt = tOpenPoints.fData.begin(); tIt != tOpenPoints.fData.end(); tIt++ )
//		{
//			vismsg( eNormal ) <<*tIt<<eom;
//		}

    	anOrderedPoints.fData.push_back( tOpenPoints );
    	return;
    }


    void KGROOTGeometryPainter::CreateDualOrderedPoints( const IntersectionPoints anIntersectionPoints, OrderedPoints& anOrderedPoints )
    {
    	OpenPoints tPoints1, tPoints2;
		for( IntersectionPoints::SetCIt tSetIt = anIntersectionPoints.fData.begin(); tSetIt != anIntersectionPoints.fData.end(); tSetIt++ )
		{
    		IntersectionPoints::NamedGroup tNamedGroup = *tSetIt;
    		IntersectionPoints::Origin tOrigin = tNamedGroup.second;
    		IntersectionPoints::Group tGroup = tNamedGroup.first;
			if ( tGroup.size() > 2 || tOrigin != IntersectionPoints::eCircle )
			{
				continue;
			}
			KTwoVector tPoint1 = tGroup.front();
			KTwoVector tPoint2 = tGroup.back();
			vismsg_debug( "group with type <"<<tOrigin<<"> has points: "<<eom);
			vismsg_debug( tPoint1<<eom);
			vismsg_debug( tPoint2<<eom);
			if ( tPoint1.X() + tPoint1.Y() > tPoint2.X() + tPoint2.Y() )
			{
				tPoints1.fData.push_back( tPoint1 );
				tPoints2.fData.push_back( tPoint2 );
			}
			else
			{
				tPoints1.fData.push_back( tPoint2 );
				tPoints2.fData.push_back( tPoint1 );
			}
		}

		anOrderedPoints.fData.push_back( tPoints1 );
		anOrderedPoints.fData.push_back( tPoints2 );
    	return;
    }

    void KGROOTGeometryPainter::CombineOrderedPoints( OrderedPoints& anOrderedPoints )
    {
        //check for two groups of points and combine
    	//this function is needed, if a space cross the z axis, like the wafer
        if ( anOrderedPoints.fData.size() == 2 )
        {
        	Points tNewPoints;
			OrderedPoints::SetCIt tSetIt = anOrderedPoints.fData.begin();
			Points tPoints = *tSetIt;
			for ( Points::CIt tGroupIt = tPoints.fData.begin(); tGroupIt != tPoints.fData.end(); tGroupIt++ )
			{
				KTwoVector tPoint = *tGroupIt;
				tNewPoints.fData.push_back( tPoint );
			}
			tSetIt++;
			tPoints = *tSetIt;
			for ( Points::Set::const_reverse_iterator tReversedGroupIt = tPoints.fData.rbegin(); tReversedGroupIt != tPoints.fData.rend(); tReversedGroupIt++ )
			{
				KTwoVector tPoint = *tReversedGroupIt;
				tNewPoints.fData.push_back( tPoint );
			}
			anOrderedPoints.fData.clear();
			anOrderedPoints.fData.push_back( tNewPoints );
        }
    }


    //*******************
    //rendering functions
    //*******************

    void KGROOTGeometryPainter::OrderedPointsToROOTSurface( const OrderedPoints anOrderedPoints)
    {
    	for ( OrderedPoints::SetCIt tSetIt = anOrderedPoints.fData.begin(); tSetIt != anOrderedPoints.fData.end(); tSetIt++ )
    	{
    		TPolyLine* tPolyLine = new TPolyLine();
    		KTwoVector tLastPoint;
    		Points tPoints = *tSetIt;
    		for ( Points::CIt tGroupIt = tPoints.fData.begin(); tGroupIt != tPoints.fData.end(); tGroupIt++ )
    		{
    			KTwoVector tPoint = *tGroupIt;
    			if ( tPoint.X() == tLastPoint.X() && tPoint.Y() == tLastPoint.Y() && tGroupIt !=tPoints.fData.begin()  )
    			{
    				//skip points that are the same
    				continue;
    			}
    			tPolyLine->SetNextPoint( tPoint.X(), tPoint.Y() );
    			tLastPoint = tPoint;
    		}
    		fROOTSurfaces.push_back( tPolyLine );
    	}
    	return;
    }


    void KGROOTGeometryPainter::OrderedPointsToROOTSpace( const OrderedPoints anOrderedPoints)
    {
    	for ( OrderedPoints::SetCIt tSetIt = anOrderedPoints.fData.begin(); tSetIt != anOrderedPoints.fData.end(); tSetIt++ )
    	{
    		TPolyLine* tPolyLine = new TPolyLine();
    		KTwoVector tLastPoint;
    		Points tPoints = *tSetIt;
    		for ( Points::CIt tGroupIt = tPoints.fData.begin(); tGroupIt != tPoints.fData.end(); tGroupIt++ )
    		{
    			KTwoVector tPoint = *tGroupIt;
    			if ( tPoint.X() == tLastPoint.X() && tPoint.Y() == tLastPoint.Y() && tGroupIt !=tPoints.fData.begin()  )
    			{
    				//skip points that are the same
    				continue;
    			}
    			tPolyLine->SetNextPoint( tPoint.X(), tPoint.Y() );
    			tLastPoint = tPoint;
    		}
    		fROOTSpaces.push_back( tPolyLine );
    	}
    	return;
    }


//    void KGROOTGeometryPainter::IntersectionsToROOT( const IntersectionPoints aCircleIntersection, const IntersectionPoints aParallelIntersection )
//    {
//    	for( IntersectionPoints::SetCIt tSetIt = aCircleIntersection.fData.begin(); tSetIt != aCircleIntersection.fData.end(); tSetIt++ )
//    	{
//    		IntersectionPoints::Group tGroup = *tSetIt;
//    		if ( tGroup.size() > 2 )
//    		{
//    			vismsg( eWarning ) <<"It looks like Stefan forgot one case in connecting the intersection points, so your drawing may look like bullshit"<<eom;
//    		}
//    	}
//
//    	//check on which side the drawing is not closed
//    	bool tFrontSideOpen = false;
//    	bool tRearSideOpen = false;;
//    	IntersectionPoints::SetCIt tCircleSetCheckIt = aCircleIntersection.fData.begin();
//    	if ( (*tCircleSetCheckIt).size() == 2 )
//    	{
//    		tFrontSideOpen = true;
//    	}
//    	tCircleSetCheckIt = aCircleIntersection.fData.end() - 1;
//    	if ( (*tCircleSetCheckIt).size() == 2 )
//    	{
//    		tRearSideOpen = true;
//    	}
//
//    	if ( tRearSideOpen && !tFrontSideOpen )
//    	{
//    		vismsg_debug( "Front closed, rear open!"<<eom);
//    		TPolyLine* tPolyLine = new TPolyLine();
//    		//start with rear circle intersection
//        	IntersectionPoints::SetCIt tCircleSetIt = aCircleIntersection.fData.end();
//
//
//
//    	}
//
//
//		TPolyLine* tPolyLine1 = new TPolyLine();
//    	TPolyLine* tPolyLine2 = new TPolyLine();
//    	tPolyLine2->SetLineColor( kRed );
//
//    	for( IntersectionPoints::SetCIt tSetIt = aParallelIntersection.fData.begin(); tSetIt != aParallelIntersection.fData.end(); tSetIt++ )
//    	{
//    		IntersectionPoints::Group tGroup = *tSetIt;
//    	   	for( IntersectionPoints::GroupCIt tGroupIt = tGroup.begin(); tGroupIt != tGroup.end(); tGroupIt++ )
//			{
//    	   		tPolyLine1->SetNextPoint( (*tGroupIt).X(), (*tGroupIt).Y() );
//			}
//    	}
//
//    	for( IntersectionPoints::SetCIt tSetIt = aCircleIntersection.fData.begin(); tSetIt != aCircleIntersection.fData.end(); tSetIt++ )
//    	{
//    		IntersectionPoints::Group tGroup = *tSetIt;
//    	   	for( IntersectionPoints::GroupCIt tGroupIt = tGroup.begin(); tGroupIt != tGroup.end(); tGroupIt++ )
//			{
//    	   		tPolyLine2->SetNextPoint( (*tGroupIt).X(), (*tGroupIt).Y() );
//			}
//    	}
//
//    	fROOTSurfaces.push_back( tPolyLine1 );
//    	fROOTSurfaces.push_back( tPolyLine2 );
//
//    	return;
//    }

//    void KGROOTGeometryPainter::PolyLinesToROOT( const PolyLines aPolyLine )
//    {
//    	for( PolyLines::SetCIt tSetIt = aPolyLine.fData.begin(); tSetIt != aPolyLine.fData.end(); tSetIt++ )
//    	{
//    		TPolyLine* tPolyLine = new TPolyLine();
//    		Points tPoints = (*tSetIt);
//    		for ( Points::CIt tPointsIt = tPoints.fData.begin(); tPointsIt != tPoints.fData.end(); tPoints++ )
//    		{
//    			KTwoVector tPoint = *tPointsIt;
//    			tPolyLine->SetNextPoint( tPoint.X(), tPoint.Y() );
//    		}
//    	}
//    }



//    void KGROOTGeometryPainter::FlatMeshToVTK( const FlatMesh& aMesh )
//    {
//        //object allocation
//        KThreeVector tPoint;
//
//        deque< vtkIdType > vMeshIdGroup;
//        deque< deque< vtkIdType > > vMeshIdSet;
//
//        deque< deque< vtkIdType > >::iterator vThisGroup;
//        deque< deque< vtkIdType > >::iterator vNextGroup;
//
//        deque< vtkIdType >::iterator vThisThisPoint;
//        deque< vtkIdType >::iterator vThisNextPoint;
//        deque< vtkIdType >::iterator vNextThisPoint;
//        deque< vtkIdType >::iterator vNextNextPoint;
//
//        vtkSmartPointer< vtkQuad > vQuad;
//
//        //create mesh point ids
//        for( TubeMesh::SetCIt tSetIt = aMesh.fData.begin(); tSetIt != aMesh.fData.end(); tSetIt++ )
//        {
//            vMeshIdGroup.clear();
//            for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
//            {
//                LocalToGlobal( *tGroupIt, tPoint );
//                vMeshIdGroup.push_back( fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() ) );
//            }
//            vMeshIdSet.push_back( vMeshIdGroup );
//        }
//
//        //create hull cells
//        vThisGroup = vMeshIdSet.begin();
//        vNextGroup = ++(vMeshIdSet.begin());
//        while( vNextGroup != vMeshIdSet.end() )
//        {
//            vThisThisPoint = vThisGroup->begin();
//            vThisNextPoint = ++(vThisGroup->begin());
//            vNextThisPoint = vNextGroup->begin();
//            vNextNextPoint = ++(vNextGroup->begin());
//
//            while( vNextNextPoint != vNextGroup->end() )
//            {
//                vQuad = vtkSmartPointer< vtkQuad >::New();
//                vQuad->GetPointIds()->SetId( 0, *vThisThisPoint );
//                vQuad->GetPointIds()->SetId( 1, *vNextThisPoint );
//                vQuad->GetPointIds()->SetId( 2, *vNextNextPoint );
//                vQuad->GetPointIds()->SetId( 3, *vThisNextPoint );
//                fCells->InsertNextCell( vQuad );
//                fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
//                ++vThisThisPoint;
//                ++vThisNextPoint;
//                ++vNextThisPoint;
//                ++vNextNextPoint;
//            }
//
//            ++vThisGroup;
//            ++vNextGroup;
//        }
//
//        return;
//    }
//    void KGROOTGeometryPainter::TubeMeshToROOT( const TubeMesh& aMesh )
//    {
//        //object allocation
//        KThreeVector tPoint, tPointSetLast, tPointGroupLast;
//
//        KThreeVector tIntersectionPoint;
//        bool tIntersection;
//
//        vector< KThreeVector> tIntersectionPoints;
//
//        //create intersection points for root
//        for( TubeMesh::SetCIt tSetIt = aMesh.fData.begin(); tSetIt != aMesh.fData.end(); tSetIt++ )
//        {
//
//            for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
//            {
//                LocalToGlobal( *tGroupIt, tPoint );
//                if ( tGroupIt == tSetIt->begin() )
//                {
//                	if ( tSetIt != aMesh.fData.begin() )
//                	{
//						CalculatePlaneIntersection( tPointSetLast, tPoint, tIntersectionPoint, tIntersection );
//	                	if ( tIntersection )
//	                	{
//	                		tIntersectionPoints.push_back( tIntersectionPoint );
//	                	}
//                	}
//                	tPointSetLast = tPoint;
//                }
//                else
//                {
//                	CalculatePlaneIntersection( tPointGroupLast, tPoint, tIntersectionPoint, tIntersection );
//                	if ( tIntersection )
//                	{
//                		tIntersectionPoints.push_back( tIntersectionPoint );
//                	}
//                }
//                if ( tGroupIt == tSetIt->end() - 1 )
//                {
//                	CalculatePlaneIntersection( tPoint, tPointSetLast, tIntersectionPoint, tIntersection );
//                	if ( tIntersection )
//                	{
//                		tIntersectionPoints.push_back( tIntersectionPoint );
//                	}
//                }
//                tPointGroupLast = tPoint;
//            }
//        }
//
//
//		if ( fPlane == eZYPlane )
//		{
//			TPolyLine* tPolyLine1 = new TPolyLine();
//			TPolyLine* tPolyLine2 = new TPolyLine();
//
//			TColor* tColor = new TColor();
//			tColor->SetRGB( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue() );
//			tPolyLine1->SetLineColor( tColor->GetNumber() );
//			tPolyLine2->SetLineColor( tColor->GetNumber() );
//
//			delete tColor;
//
//			vector<KThreeVector>::iterator tIt;
//	        for ( tIt = tIntersectionPoints.begin(); tIt != tIntersectionPoints.end(); tIt++ )
//	        {
//	        	tPolyLine1->SetNextPoint( (*tIt).Z(), (*tIt).Y() );
//	        	tIt++;
//	        	tPolyLine2->SetNextPoint( (*tIt).Z(), (*tIt).Y() );
//	        }
//	        fROOTSurfaces.push_back( tPolyLine1 );
//	        fROOTSurfaces.push_back( tPolyLine2 );
//		}
//
//		if ( fPlane == eZXPlane )
//		{
//			TPolyLine* tPolyLine1 = new TPolyLine();
//			TPolyLine* tPolyLine2 = new TPolyLine();
//
//			TColor* tColor = new TColor();
//			tColor->SetRGB( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue() );
//			tPolyLine1->SetLineColor( tColor->GetNumber() );
//			tPolyLine2->SetLineColor( tColor->GetNumber() );
//
//			delete tColor;
//
//			vector<KThreeVector>::iterator tIt;
//	        for ( tIt = tIntersectionPoints.begin(); tIt != tIntersectionPoints.end(); tIt++ )
//	        {
//	        	tPolyLine1->SetNextPoint( (*tIt).Z(), (*tIt).X() );
//	        	tIt++;
//	        	tPolyLine2->SetNextPoint( (*tIt).Z(), (*tIt).X() );
//	        }
//	        fROOTSurfaces.push_back( tPolyLine1 );
//	        fROOTSurfaces.push_back( tPolyLine2 );
//		}
//
//		if ( fPlane == eXYPlane )
//		{
//			TPolyLine* tPolyLine = new TPolyLine();
//
//			TColor* tColor = new TColor();
//			tColor->SetRGB( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue() );
//			tPolyLine->SetLineColor( tColor->GetNumber() );
//
//			delete tColor;
//
//			vector<KThreeVector>::iterator tIt;
//	        for ( tIt = tIntersectionPoints.begin(); tIt != tIntersectionPoints.end(); tIt++ )
//	        {
//	        	tPolyLine->SetNextPoint( (*tIt).Z(), (*tIt).X() );
//	        }
//	        fROOTSurfaces.push_back( tPolyLine );
//		}
//
//
//
//        return;
//    }
//    void KGVTKGeometryPainter::TubeMeshToVTK( const KThreeVector& anApexStart, const TubeMesh& aMesh )
//    {
//        //object allocation
//        KThreeVector tPoint;
//
//        vtkIdType vMeshIdApexStart;
//        deque< vtkIdType > vMeshIdGroup;
//        deque< deque< vtkIdType > > vMeshIdSet;
//
//        deque< deque< vtkIdType > >::iterator vThisGroup;
//        deque< deque< vtkIdType > >::iterator vNextGroup;
//
//        deque< vtkIdType >::iterator vThisThisPoint;
//        deque< vtkIdType >::iterator vThisNextPoint;
//        deque< vtkIdType >::iterator vNextThisPoint;
//        deque< vtkIdType >::iterator vNextNextPoint;
//
//        vtkSmartPointer< vtkTriangle > vTriangle;
//        vtkSmartPointer< vtkQuad > vQuad;
//
//        //create apex start point id
//        LocalToGlobal( anApexStart, tPoint );
//        vMeshIdApexStart = fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() );
//
//        //create mesh point ids
//        for( TubeMesh::SetCIt tSetIt = aMesh.fData.begin(); tSetIt != aMesh.fData.end(); tSetIt++ )
//        {
//            vMeshIdGroup.clear();
//            for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
//            {
//                LocalToGlobal( *tGroupIt, tPoint );
//                vMeshIdGroup.push_back( fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() ) );
//            }
//            vMeshIdGroup.push_back( vMeshIdGroup.front() );
//            vMeshIdSet.push_back( vMeshIdGroup );
//        }
//
//        //create start cap cells
//        vThisThisPoint = vMeshIdSet.front().begin();
//        vThisNextPoint = ++(vMeshIdSet.front().begin());
//        while( vThisNextPoint != vMeshIdSet.front().end() )
//        {
//            vTriangle = vtkSmartPointer< vtkTriangle >::New();
//            vTriangle->GetPointIds()->SetId( 0, *vThisThisPoint );
//            vTriangle->GetPointIds()->SetId( 1, *vThisNextPoint );
//            vTriangle->GetPointIds()->SetId( 2, vMeshIdApexStart );
//            fCells->InsertNextCell( vTriangle );
//            fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
//            ++vThisThisPoint;
//            ++vThisNextPoint;
//        }
//
//        //create hull cells
//        vThisGroup = vMeshIdSet.begin();
//        vNextGroup = ++(vMeshIdSet.begin());
//        while( vNextGroup != vMeshIdSet.end() )
//        {
//            vThisThisPoint = vThisGroup->begin();
//            vThisNextPoint = ++(vThisGroup->begin());
//            vNextThisPoint = vNextGroup->begin();
//            vNextNextPoint = ++(vNextGroup->begin());
//
//            while( vNextNextPoint != vNextGroup->end() )
//            {
//                vQuad = vtkSmartPointer< vtkQuad >::New();
//                vQuad->GetPointIds()->SetId( 0, *vThisThisPoint );
//                vQuad->GetPointIds()->SetId( 1, *vNextThisPoint );
//                vQuad->GetPointIds()->SetId( 2, *vNextNextPoint );
//                vQuad->GetPointIds()->SetId( 3, *vThisNextPoint );
//                fCells->InsertNextCell( vQuad );
//                fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
//                ++vThisThisPoint;
//                ++vThisNextPoint;
//                ++vNextThisPoint;
//                ++vNextNextPoint;
//            }
//
//            ++vThisGroup;
//            ++vNextGroup;
//        }
//
//        return;
//    }
//    void KGVTKGeometryPainter::TubeMeshToVTK( const TubeMesh& aMesh, const KThreeVector& anApexEnd )
//    {
//        //object allocation
//        KThreeVector tPoint;
//
//        vtkIdType vMeshIdApexEnd;
//        deque< vtkIdType > vMeshIdGroup;
//        deque< deque< vtkIdType > > vMeshIdSet;
//
//        deque< deque< vtkIdType > >::iterator vThisGroup;
//        deque< deque< vtkIdType > >::iterator vNextGroup;
//
//        deque< vtkIdType >::iterator vThisThisPoint;
//        deque< vtkIdType >::iterator vThisNextPoint;
//        deque< vtkIdType >::iterator vNextThisPoint;
//        deque< vtkIdType >::iterator vNextNextPoint;
//
//        vtkSmartPointer< vtkTriangle > vTriangle;
//        vtkSmartPointer< vtkQuad > vQuad;
//
//        //create apex end point id
//        LocalToGlobal( anApexEnd, tPoint );
//        vMeshIdApexEnd = fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() );
//
//        //create mesh point ids
//        for( TubeMesh::SetCIt tSetIt = aMesh.fData.begin(); tSetIt != aMesh.fData.end(); tSetIt++ )
//        {
//            vMeshIdGroup.clear();
//            for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
//            {
//                LocalToGlobal( *tGroupIt, tPoint );
//                vMeshIdGroup.push_back( fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() ) );
//            }
//            vMeshIdGroup.push_back( vMeshIdGroup.front() );
//            vMeshIdSet.push_back( vMeshIdGroup );
//        }
//
//        //create hull cells
//        vThisGroup = vMeshIdSet.begin();
//        vNextGroup = ++(vMeshIdSet.begin());
//        while( vNextGroup != vMeshIdSet.end() )
//        {
//            vThisThisPoint = vThisGroup->begin();
//            vThisNextPoint = ++(vThisGroup->begin());
//            vNextThisPoint = vNextGroup->begin();
//            vNextNextPoint = ++(vNextGroup->begin());
//
//            while( vNextNextPoint != vNextGroup->end() )
//            {
//                vQuad = vtkSmartPointer< vtkQuad >::New();
//                vQuad->GetPointIds()->SetId( 0, *vThisThisPoint );
//                vQuad->GetPointIds()->SetId( 1, *vNextThisPoint );
//                vQuad->GetPointIds()->SetId( 2, *vNextNextPoint );
//                vQuad->GetPointIds()->SetId( 3, *vThisNextPoint );
//                fCells->InsertNextCell( vQuad );
//                fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
//                ++vThisThisPoint;
//                ++vThisNextPoint;
//                ++vNextThisPoint;
//                ++vNextNextPoint;
//            }
//
//            ++vThisGroup;
//            ++vNextGroup;
//        }
//
//        //create end cap cells
//        vThisThisPoint = vMeshIdSet.back().begin();
//        vThisNextPoint = ++(vMeshIdSet.back().begin());
//        while( vThisNextPoint != vMeshIdSet.back().end() )
//        {
//            vTriangle = vtkSmartPointer< vtkTriangle >::New();
//            vTriangle->GetPointIds()->SetId( 0, *vThisThisPoint );
//            vTriangle->GetPointIds()->SetId( 1, *vThisNextPoint );
//            vTriangle->GetPointIds()->SetId( 2, vMeshIdApexEnd );
//            fCells->InsertNextCell( vTriangle );
//            fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
//            ++vThisThisPoint;
//            ++vThisNextPoint;
//        }
//
//        return;
//    }
//    void KGVTKGeometryPainter::TubeMeshToVTK( const KThreeVector& anApexStart, const TubeMesh& aMesh, const KThreeVector& anApexEnd )
//    {
//        //object allocation
//        KThreeVector tPoint;
//
//        vtkIdType vMeshIdApexStart;
//        vtkIdType vMeshIdApexEnd;
//        deque< vtkIdType > vMeshIdGroup;
//        deque< deque< vtkIdType > > vMeshIdSet;
//
//        deque< deque< vtkIdType > >::iterator vThisGroup;
//        deque< deque< vtkIdType > >::iterator vNextGroup;
//
//        deque< vtkIdType >::iterator vThisThisPoint;
//        deque< vtkIdType >::iterator vThisNextPoint;
//        deque< vtkIdType >::iterator vNextThisPoint;
//        deque< vtkIdType >::iterator vNextNextPoint;
//
//        vtkSmartPointer< vtkTriangle > vTriangle;
//        vtkSmartPointer< vtkQuad > vQuad;
//
//        //create apex start point id
//        LocalToGlobal( anApexStart, tPoint );
//        vMeshIdApexStart = fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() );
//
//        //create apex end point id
//        LocalToGlobal( anApexEnd, tPoint );
//        vMeshIdApexEnd = fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() );
//
//        //create mesh point ids
//        for( TubeMesh::SetCIt tSetIt = aMesh.fData.begin(); tSetIt != aMesh.fData.end(); tSetIt++ )
//        {
//            vMeshIdGroup.clear();
//            for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
//            {
//                LocalToGlobal( *tGroupIt, tPoint );
//                vMeshIdGroup.push_back( fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() ) );
//            }
//            vMeshIdGroup.push_back( vMeshIdGroup.front() );
//            vMeshIdSet.push_back( vMeshIdGroup );
//        }
//
//        //create start cap cells
//        vThisThisPoint = vMeshIdSet.front().begin();
//        vThisNextPoint = ++(vMeshIdSet.front().begin());
//        while( vThisNextPoint != vMeshIdSet.front().end() )
//        {
//            vTriangle = vtkSmartPointer< vtkTriangle >::New();
//            vTriangle->GetPointIds()->SetId( 0, *vThisThisPoint );
//            vTriangle->GetPointIds()->SetId( 1, *vThisNextPoint );
//            vTriangle->GetPointIds()->SetId( 2, vMeshIdApexStart );
//            fCells->InsertNextCell( vTriangle );
//            fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
//            ++vThisThisPoint;
//            ++vThisNextPoint;
//        }
//
//        //create hull cells
//        vThisGroup = vMeshIdSet.begin();
//        vNextGroup = ++(vMeshIdSet.begin());
//        while( vNextGroup != vMeshIdSet.end() )
//        {
//            vThisThisPoint = vThisGroup->begin();
//            vThisNextPoint = ++(vThisGroup->begin());
//            vNextThisPoint = vNextGroup->begin();
//            vNextNextPoint = ++(vNextGroup->begin());
//
//            while( vNextNextPoint != vNextGroup->end() )
//            {
//                vQuad = vtkSmartPointer< vtkQuad >::New();
//                vQuad->GetPointIds()->SetId( 0, *vThisThisPoint );
//                vQuad->GetPointIds()->SetId( 1, *vNextThisPoint );
//                vQuad->GetPointIds()->SetId( 2, *vNextNextPoint );
//                vQuad->GetPointIds()->SetId( 3, *vThisNextPoint );
//                fCells->InsertNextCell( vQuad );
//                fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
//                ++vThisThisPoint;
//                ++vThisNextPoint;
//                ++vNextThisPoint;
//                ++vNextNextPoint;
//            }
//
//            ++vThisGroup;
//            ++vNextGroup;
//        }
//
//        //create end cap cells
//        vThisThisPoint = vMeshIdSet.back().begin();
//        vThisNextPoint = ++(vMeshIdSet.back().begin());
//        while( vThisNextPoint != vMeshIdSet.back().end() )
//        {
//            vTriangle = vtkSmartPointer< vtkTriangle >::New();
//            vTriangle->GetPointIds()->SetId( 0, *vThisThisPoint );
//            vTriangle->GetPointIds()->SetId( 1, *vThisNextPoint );
//            vTriangle->GetPointIds()->SetId( 2, vMeshIdApexEnd );
//            fCells->InsertNextCell( vTriangle );
//            fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
//            ++vThisThisPoint;
//            ++vThisNextPoint;
//        }
//
//        return;
//    }
//    void KGVTKGeometryPainter::TorusMeshToVTK( const TorusMesh& aMesh )
//    {
//        //object allocation
//        KThreeVector tPoint;
//
//        deque< vtkIdType > vMeshIdGroup;
//        deque< deque< vtkIdType > > vMeshIdSet;
//
//        deque< deque< vtkIdType > >::iterator vThisGroup;
//        deque< deque< vtkIdType > >::iterator vNextGroup;
//
//        deque< vtkIdType >::iterator vThisThisPoint;
//        deque< vtkIdType >::iterator vThisNextPoint;
//        deque< vtkIdType >::iterator vNextThisPoint;
//        deque< vtkIdType >::iterator vNextNextPoint;
//
//        vtkSmartPointer< vtkQuad > vQuad;
//
//        //create mesh point ids
//        for( TubeMesh::SetCIt tSetIt = aMesh.fData.begin(); tSetIt != aMesh.fData.end(); tSetIt++ )
//        {
//            vMeshIdGroup.clear();
//            for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
//            {
//                LocalToGlobal( *tGroupIt, tPoint );
//                vMeshIdGroup.push_back( fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() ) );
//            }
//            vMeshIdGroup.push_back( vMeshIdGroup.front() );
//            vMeshIdSet.push_back( vMeshIdGroup );
//        }
//        vMeshIdSet.push_back( vMeshIdSet.front() );
//
//        //create hull cells
//        vThisGroup = vMeshIdSet.begin();
//        vNextGroup = ++(vMeshIdSet.begin());
//        while( vNextGroup != vMeshIdSet.end() )
//        {
//            vThisThisPoint = vThisGroup->begin();
//            vThisNextPoint = ++(vThisGroup->begin());
//            vNextThisPoint = vNextGroup->begin();
//            vNextNextPoint = ++(vNextGroup->begin());
//
//            while( vNextNextPoint != vNextGroup->end() )
//            {
//                vQuad = vtkSmartPointer< vtkQuad >::New();
//                vQuad->GetPointIds()->SetId( 0, *vThisThisPoint );
//                vQuad->GetPointIds()->SetId( 1, *vNextThisPoint );
//                vQuad->GetPointIds()->SetId( 2, *vNextNextPoint );
//                vQuad->GetPointIds()->SetId( 3, *vThisNextPoint );
//                fCells->InsertNextCell( vQuad );
//                fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
//                ++vThisThisPoint;
//                ++vThisNextPoint;
//                ++vNextThisPoint;
//                ++vNextNextPoint;
//            }
//
//            ++vThisGroup;
//            ++vNextGroup;
//        }
//
//        return;
//    }

}
