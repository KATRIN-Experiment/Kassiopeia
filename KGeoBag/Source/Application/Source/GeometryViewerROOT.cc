#include "KGCoreMessage.hh"
#include "KGInterfaceBuilder.hh"
#include "KGROOTGeometryPainter.hh"
#include "KROOTWindow.h"

#include "KMessage.h"
#include "KTextFile.h"
#include "KCommandLineTokenizer.hh"
#include "KXMLTokenizer.hh"
#include "KVariableProcessor.hh"
#include "KFormulaProcessor.hh"
#include "KIncludeProcessor.hh"
#include "KLoopProcessor.hh"
#include "KConditionProcessor.hh"
#include "KPrintProcessor.hh"
#include "KElementProcessor.hh"
#include "KTagProcessor.hh"

using namespace KGeoBag;
using namespace katrin;
using namespace std;

int main( int argc, char** argv )
{
    if( argc < 3 )
    {
        cout << "usage: ./GeometryPainterROOT <config_file_name.xml> <geometry_path> <plane normal vector> <plane point> <swap axis>" << endl;
        return -1;
    }

    KMessageTable::GetInstance().SetTerminalVerbosity( eDebug );
    KMessageTable::GetInstance().SetLogVerbosity( eDebug );
    KMessageTable::GetInstance().Get( "k_initialization" )->SetTerminalVerbosity( eNormal );

    string tFileName( argv[ 1 ] );
    string tPath( argv[ 2 ] );

    KCommandLineTokenizer tCommandLine;
    tCommandLine.ProcessCommandLine( argc, argv );

    KXMLTokenizer tTokenizer;
    KVariableProcessor tVariableProcessor( tCommandLine.GetVariables() );
    KFormulaProcessor tFormulaProcessor;
    KIncludeProcessor tIncludeProcessor;
    KLoopProcessor tLoopProcessor;
    KConditionProcessor tConditionProcessor;
    KPrintProcessor tPrintProcessor;
    KTagProcessor tTagProcessor;
    KElementProcessor tElementProcessor;

    tVariableProcessor.InsertAfter( &tTokenizer );
    tFormulaProcessor.InsertAfter( &tVariableProcessor );
    tIncludeProcessor.InsertAfter( &tFormulaProcessor );
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

    KROOTWindow tWindow;
    tWindow.SetName( "KGeoBag ROOT Geometry Viewer" );

    KGROOTGeometryPainter tPainter;
    tPainter.SetName( "ROOT GeometryPainter" );
    tPainter.SetDisplayMode( true );
    tPainter.SetWriteMode( true );

    if ( argc >= 6 )
    {
    	string tNormalX( argv[ 3 ] );
    	string tNormalY( argv[ 4 ] );
    	string tNormalZ( argv[ 5 ] );
    	string tSpaceString(" ");
    	string tCombine = tNormalX+tSpaceString+tNormalY+tSpaceString+tNormalZ;
        istringstream Converter( tCombine );
    	KThreeVector tPlaneNormal;
        Converter >> tPlaneNormal;
        tPainter.SetPlaneNormal( tPlaneNormal );
    }

    if ( argc >= 9 )
    {
    	string tX( argv[ 6 ] );
    	string tY( argv[ 7 ] );
    	string tZ( argv[ 8 ] );
    	string tSpaceString(" ");
    	string tCombine = tX+tSpaceString+tY+tSpaceString+tZ;
        istringstream Converter( tCombine );
    	KThreeVector tPlanePoint;
        Converter >> tPlanePoint;
        tPainter.SetPlanePoint( tPlanePoint );
    }

    if ( argc == 10 )
    {
    	if ( argv[ 9 ] == string("true"))
    	{
    		tPainter.SetSwapAxis( true );
    	}
    	if ( argv[ 9 ] == string("false"))
    	{
    		tPainter.SetSwapAxis( false );
    	}
    }

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
    tWindow.Display();
    tWindow.Write();
    tWindow.RemovePainter( &tPainter );

    return 0;
}
