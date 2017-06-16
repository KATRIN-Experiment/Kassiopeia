#include "KGCoreMessage.hh"
#include "KGInterfaceBuilder.hh"
#include "KGVTKGeometryPainter.hh"

#include "KCommandLineTokenizer.hh"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KConditionProcessor.hh"
#include "KPrintProcessor.hh"
#include "KElementProcessor.hh"
#include "KTagProcessor.hh"
#include "KMessage.h"
#include "KTextFile.h"
#include "KVTKWindow.h"

#ifdef KGeoBag_USE_ROOT
#include "KFormulaProcessor.hh"
#endif

using namespace KGeoBag;
using namespace katrin;
using namespace std;

int main( int argc, char** argv )
{
    if( argc < 3 )
    {
        cout << "usage: ./GeometryViewer <config_file_name.xml> <geometry_path>" << endl;
        return -1;
    }

    KMessageTable::GetInstance().SetTerminalVerbosity( eDebug );
    KMessageTable::GetInstance().SetLogVerbosity( eDebug );

    string tFileName( argv[ 1 ] );
    string tPath( argv[ 2 ] );

    KCommandLineTokenizer tCommandLine;
    tCommandLine.ProcessCommandLine( argc, argv );

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor( tCommandLine.GetVariables() );
    KIncludeProcessor tIncludeProcessor;
    KLoopProcessor tLoopProcessor;
    KConditionProcessor tConditionProcessor;
    KPrintProcessor tPrintProcessor;
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
    tWindow.SetName( "KGeoBag Geometry Viewer" );
    tWindow.SetFrameColorRed( 0. );
    tWindow.SetFrameColorGreen( 0. );
    tWindow.SetFrameColorBlue( 0. );
    tWindow.SetDisplayMode( true );
    tWindow.SetWriteMode( true );

    KGVTKGeometryPainter tPainter;
    tPainter.SetName( "GeometryPainter" );
    tPainter.SetDisplayMode( true );
    tPainter.SetWriteMode( true );

    vector< KGSurface* > tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces( tPath );
    vector< KGSurface* >::iterator tSurfaceIt;

    vector< KGSpace* > tSpaces = KGInterface::GetInstance()->RetrieveSpaces( tPath );
    vector< KGSpace* >::iterator tSpaceIt;

    for( tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++ )
    {
        tPainter.AddSurface( *tSurfaceIt );
    }
    for( tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++ )
    {
        tPainter.AddSpace( *tSpaceIt );
    }

    tWindow.AddPainter( &tPainter );
    tWindow.Render();
    tWindow.Write();
    tWindow.Display();
    tWindow.RemovePainter( &tPainter );

    return 0;
}
