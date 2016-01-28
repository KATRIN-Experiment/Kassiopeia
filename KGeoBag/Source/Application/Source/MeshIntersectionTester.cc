#include "KGCoreMessage.hh"
#include "KGInterfaceBuilder.hh"
#include "KGMesher.hh"
#include "KGVTKMeshPainter.hh"

#include "KMessage.h"
#include "KTextFile.h"
#include "KCommandLineTokenizer.hh"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KConditionProcessor.hh"
#include "KPrintProcessor.hh"
#include "KElementProcessor.hh"
#include "KTagProcessor.hh"
#include "KVTKWindow.h"

#include "KGVTKMeshIntersectionTester.hh"


#ifdef Kommon_USE_ROOT
#include "KFormulaProcessor.hh"
#endif

using namespace KGeoBag;
using namespace katrin;

int main( int argc, char** argv )
{
    if( argc < 3 )
    {
        cout << "usage: ./MeshViewer <config_file_name.xml> <geometry_path>" << endl;
        return -1;
    }

    string tFileName( argv[ 1 ] );
    string tPath( argv[ 2 ] );

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor;
    KIncludeProcessor tIncludeProcessor;
    KLoopProcessor tLoopProcessor;
    KConditionProcessor tConditionProcessor;
    KPrintProcessor tPrintProcessor;
    KTagProcessor tTagProcessor;
    KElementProcessor tElementProcessor;

    tVariableProcessor.InsertAfter( &tTokenizer );
    tIncludeProcessor.InsertAfter( &tVariableProcessor );

  #ifdef Kommon_USE_ROOT
  	KFormulaProcessor tFormulaProcessor;
  	tFormulaProcessor.InsertAfter( &tVariableProcessor );
  	tIncludeProcessor.InsertAfter( &tFormulaProcessor );
  #endif

    tLoopProcessor.InsertAfter( &tIncludeProcessor );
    tConditionProcessor.InsertAfter( &tLoopProcessor );
    tPrintProcessor.InsertAfter( &tConditionProcessor );
    tTagProcessor.InsertAfter( &tPrintProcessor );
    tElementProcessor.InsertAfter( &tTagProcessor );

    coremsg( eNormal ) << "starting initialization..." << eom;

    KTextFile tFile;
    tFile.SetDefaultPath( CONFIG_DEFAULT_DIR );
    tFile.AddToNames( tFileName );
    tTokenizer.ProcessFile( &tFile );

    coremsg( eNormal ) << "...initialization finished" << eom;

    KVTKWindow tWindow;
    tWindow.SetName( "KGeoBag Mesh Viewer" );
    tWindow.SetFrameColorRed( 0. );
    tWindow.SetFrameColorGreen( 0. );
    tWindow.SetFrameColorBlue( 0. );
    tWindow.SetDisplayMode( true );
    tWindow.SetWriteMode( true );

    KGMesher tMesher;

    KGVTKMeshPainter tPainter;
    tPainter.SetName( "MeshPainter" );
    tPainter.SetDisplayMode( true );
    tPainter.SetWriteMode( true );
    tPainter.SetColorMode( KGVTKMeshPainter::sModulo );

    KGVTKMeshIntersectionTester tTester;
    tTester.SetName( "MeshIntersectionTester" );
    tTester.SetDisplayMode( true );
    tTester.SetWriteMode( true );
    tTester.SetSampleCount( 500 );
    tTester.SetSampleColor( KGRGBColor( 127, 127, 127 ) );
    tTester.SetUnintersectedLineColor( KGRGBColor( 127, 127, 127 ) );
    tTester.SetIntersectedLineColor( KGRGBColor( 0, 0, 255 ) );
    tTester.SetVertexSize( 0.001 );
    tTester.SetLineSize( 0.001 );

    vector< KGSurface* > tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces( tPath );
    vector< KGSurface* >::iterator tSurfaceIt;

    vector< KGSpace* > tSpaces = KGInterface::GetInstance()->RetrieveSpaces( tPath );
    vector< KGSpace* >::iterator tSpaceIt;

    for( tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++ )
    {
        (*tSurfaceIt)->MakeExtension< KGMesh >();
        (*tSurfaceIt)->AcceptNode( &tMesher );
        (*tSurfaceIt)->AcceptNode( &tPainter );
        tTester.AddSurface( *tSurfaceIt );
    }

    for( tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++ )
    {
        (*tSpaceIt)->MakeExtension< KGMesh >();
        (*tSpaceIt)->AcceptNode( &tMesher );
        (*tSpaceIt)->AcceptNode( &tPainter );
        tTester.AddSpace( *tSpaceIt );
    }

    tWindow.AddPainter( &tPainter );
    tWindow.AddPainter( &tTester );
    tWindow.Render();
    tWindow.Write();
    tWindow.Display();
    tWindow.RemovePainter( &tPainter );
    tWindow.RemovePainter( &tTester );

    return 0;
}
