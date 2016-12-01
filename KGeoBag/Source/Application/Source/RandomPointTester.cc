#include "KGCoreMessage.hh"
#include "KGInterfaceBuilder.hh"
#include "KGVTKGeometryPainter.hh"
#include "KGVTKRandomPointTester.hh"

#include "KMessage.h"
#include "KTextFile.h"
#include "KCommandLineTokenizer.hh"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KConditionProcessor.hh"
#include "KElementProcessor.hh"
#include "KTagProcessor.hh"
#include "KVTKWindow.h"

#ifdef KGeoBag_USE_ROOT
#include "KFormulaProcessor.hh"
#endif

#include "KGRandomPointGenerator.hh"

using namespace KGeoBag;
using namespace katrin;
using namespace std;

int main( int argc, char** argv )
{
    if( argc < 3 )
    {
        cout << "usage: ./NormalTester <config_file_name.xml> <geometry_path>" << endl;
        return -1;
    }

    string tFileName( argv[ 1 ] );
    string tPath( argv[ 2 ] );

    unsigned int tSampleCount = 100000;
    bool showWindow = true;

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor;
    KIncludeProcessor tIncludeProcessor;
    KLoopProcessor tLoopProcessor;
    KConditionProcessor tConditionProcessor;
    KTagProcessor tTagProcessor;
    KElementProcessor tElementProcessor;

    tVariableProcessor.InsertAfter( &tTokenizer );
    tIncludeProcessor.InsertAfter( &tVariableProcessor );

  #ifdef KGeoBag_USE_ROOT
  	KFormulaProcessor tFormulaProcessor;
  	tFormulaProcessor.InsertAfter( &tVariableProcessor );
  	tIncludeProcessor.InsertAfter( &tFormulaProcessor );
  #endif

    tLoopProcessor.InsertAfter( &tIncludeProcessor );
    tConditionProcessor.InsertAfter( &tLoopProcessor );
    tTagProcessor.InsertAfter( &tConditionProcessor );
    tElementProcessor.InsertAfter( &tTagProcessor );

    coremsg( eNormal ) << "starting initialization..." << eom;

    KTextFile tFile;
    tFile.SetDefaultPath( SCRATCH_DEFAULT_DIR );
    tFile.AddToNames( tFileName );
    tTokenizer.ProcessFile( &tFile );

    coremsg( eNormal ) << "...initialization finished" << eom;

    KVTKWindow tWindow;
    tWindow.SetName( "KGeoBag Geometry Viewer" );
    tWindow.SetFrameColorRed( 0. );
    tWindow.SetFrameColorGreen( 0. );
    tWindow.SetFrameColorBlue( 0. );
    tWindow.SetDisplayMode( true );
    tWindow.SetWriteMode( true );

    KGVTKGeometryPainter tPainter;
    tPainter.SetName( "RandomPointViewer" );
    tPainter.SetDisplayMode( true );
    tPainter.SetWriteMode( true );

    KGVTKRandomPointTester tTester;
    tTester.SetName( "RandomPointTester" );
    tTester.SetDisplayMode( true );
    tTester.SetWriteMode( true );
    tTester.SetSampleColor( KGRGBColor( 0, 255, 0 ) );
    tTester.SetVertexSize( 0.001 );

    vector< KGSurface* > tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces( tPath );
    vector< KGSurface* >::iterator tSurfaceIt;

    vector< KGSpace* > tSpaces = KGInterface::GetInstance()->RetrieveSpaces( tPath );
    vector< KGSpace* >::iterator tSpaceIt;

    for( tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++ )
    {
        tPainter.AddSurface( *tSurfaceIt );
        tTester.AddSurface( *tSurfaceIt );
    }
    for( tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++ )
    {
        tPainter.AddSpace( *tSpaceIt );
        tTester.AddSpace( *tSpaceIt );
    }

    coremsg( eNormal ) << "starting calculation of points (" << tSampleCount << ")..." << eom;
    KGRandomPointGenerator random;
    vector<KThreeVector*> tPoints;

    for( unsigned int i = 0; i < tSampleCount; ++i ) {
    	tPoints.push_back( new KThreeVector( random.Random(tSpaces) ) );
    }
    coremsg( eNormal ) << "...calculation of points finished" << eom;

    coremsg( eNormal ) << "starting calculation of points per volume..." << eom;
    vector<unsigned int> tCounter;

    for( vector<KGSpace*>::const_iterator s = tSpaces.begin();
			s != tSpaces.end(); ++s) {
		tCounter.push_back(0);
	}

    for( vector<KThreeVector*>::const_iterator p = tPoints.begin();
    		p != tPoints.end(); ++p) {
    	unsigned int c = 0;
    	for( vector<KGSpace*>::const_iterator s = tSpaces.begin();
    			s != tSpaces.end(); ++s, ++c) {
    		if(!(*s)->Outside(**p)) {
    			tCounter[c]++;
    			break;
    		}
    	}
    }
    coremsg( eNormal ) << "...calculation of points per volume finished:" << eom;
    unsigned int c = 0;
    unsigned int tTotalPoints = 0;
	for( vector<KGSpace*>::const_iterator s = tSpaces.begin();
			s != tSpaces.end(); ++s, ++c) {
		coremsg( eNormal ) << "   <" << (*s)->GetName() << ">: V = "
				<< (*s)->AsExtension<KGMetrics>()->GetVolume() << " m^3; "
				<< "points = " << tCounter[c] << "; "
				<< "density = " << (double(tCounter[c]) / (*s)->AsExtension<KGMetrics>()->GetVolume()) << " points / m^3" << eom;

		tTotalPoints += tCounter[c];
	}
	coremsg( eNormal ) << "   total points = " << tTotalPoints << eom;

	if(showWindow) {
		tTester.SetSamplePoints( tPoints );
		tWindow.AddPainter( &tPainter );
		tWindow.AddPainter( &tTester );
		tWindow.Render();
		tWindow.Write();
		tWindow.Display();
		tWindow.RemovePainter( &tPainter );
		tWindow.RemovePainter( &tTester );
	}

    return 0;
}
