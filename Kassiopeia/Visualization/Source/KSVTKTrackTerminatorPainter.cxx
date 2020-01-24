#include "KSVTKTrackTerminatorPainter.h"
#include "KSVisualizationMessage.h"

#include "KSReadFileROOT.h"
#include "KRootFile.h"

#include "vtkCellData.h"
#include "vtkPolyLine.h"
#include "vtkProperty.h"

#include <algorithm>

namespace Kassiopeia
{
    KSVTKTrackTerminatorPainter::KSVTKTrackTerminatorPainter() :
            fPointSize( 2 ),
            fPoints( vtkSmartPointer< vtkPoints >::New() ),
            fVertices( vtkSmartPointer< vtkCellArray >::New() ),
            fColors( vtkSmartPointer< vtkUnsignedCharArray >::New() ),
            fData( vtkSmartPointer< vtkPolyData >::New() ),
            fMapper( vtkSmartPointer< vtkPolyDataMapper >::New() ),
            fActor( vtkSmartPointer< vtkActor >::New() ),
            fNamedColors( vtkSmartPointer< vtkNamedColors >::New() )
    {
        fColors->SetNumberOfComponents( 3 );
        fData->SetPoints( fPoints );
        fData->SetVerts( fVertices );
        fData->GetCellData()->SetScalars( fColors );
#ifdef VTK6
        fMapper->SetInputData( fData );
#else
        fMapper->SetInput( fData );
#endif
        fMapper->SetScalarModeToUseCellData();
        fActor->SetMapper( fMapper );
    }
    KSVTKTrackTerminatorPainter::~KSVTKTrackTerminatorPainter()
    {
    }

    void KSVTKTrackTerminatorPainter::Render()
    {
        KRootFile* tRootFile = KRootFile::CreateOutputRootFile( fFile );
        if( !fPath.empty() )
        {
            tRootFile->AddToPaths( fPath );
        }

        KSReadFileROOT tReader;
        if (! tReader.TryFile( tRootFile ))
        {
            vismsg( eWarning ) << "Could not open file <" << tRootFile->GetName() << ">" << eom;
            return;
        }

        tReader.OpenFile( tRootFile );

        KSReadRunROOT& tRun = tReader.GetRun();
        KSReadEventROOT& tEvent = tReader.GetEvent();
        KSReadTrackROOT& tTrack = tReader.GetTrack();

        KSReadObjectROOT& tPointObject = tTrack.GetObject( fPointObject );
        const KSThreeVector& tPointVariable = tPointObject.Get< KSThreeVector >( fPointVariable );

        KSReadObjectROOT& tTerminatorObject = tTrack.GetObject( fTerminatorObject );
        const KSString& tTerminatorVariable = tPointObject.Get< KSString >( fTerminatorVariable );

        // TODO: Use it or delete it. bool tActive;
        vtkIdType tId;

        tRun << 0;
        for( unsigned int tEventIndex = tRun.GetFirstEventIndex(); tEventIndex <= tRun.GetLastEventIndex(); tEventIndex++ )
        {
            tEvent << tEventIndex;
            for( long tTrackIndex = tEvent.GetFirstTrackIndex(); tTrackIndex <= tEvent.GetLastTrackIndex(); tTrackIndex++ )
            {
                tTrack << tTrackIndex;

                if( (tPointObject.Valid() == true) && (tTerminatorObject.Valid() == true) )
                {
                    string tTerminatorName = tTerminatorVariable.Value();

                    if( ! fTerminators.empty() )
                    {
                        if( std::find( fTerminators.begin(), fTerminators.end(), tTerminatorName ) == fTerminators.end() )
                        {
                            vismsg_debug( "track terminator painter <" << GetName() << "> skipped track terminated by <" << tTerminatorName << ">" << eom );
                            continue;  // terminator does not match any list item
                        }
                    }

                    tId = fPoints->InsertNextPoint( tPointVariable.Value().X(), tPointVariable.Value().Y(), tPointVariable.Value().Z() );
                    fVertices->InsertNextCell( 1, &tId );

                    unsigned char red, green, blue, alpha;
                    fNamedColors->GetColor( tTerminatorName, red, green, blue, alpha );  // returns black if color is not found
                    fColors->InsertNextTuple3( red, green, blue );
                }
            }
        }

        tReader.CloseFile();

        delete tRootFile;

        fMapper->Update();

        return;
    }

    void KSVTKTrackTerminatorPainter::Display()
    {
        if( fDisplayEnabled == true )
        {
            fActor->GetProperty()->SetPointSize( fPointSize );

            vtkSmartPointer< vtkRenderer > vRenderer = fWindow->GetRenderer();
            vRenderer->AddActor( fActor );
        }

        return;
    }

    void KSVTKTrackTerminatorPainter::Write()
    {
        if( fWriteEnabled == true )
        {
            string tFile;

            if( fOutFile.length() > 0 )
            {
                if( !fPath.empty() )
                {
                    tFile = string( fPath ) + string( "/" ) + fOutFile;
                }
                else
                {
                    tFile = string( OUTPUT_DEFAULT_DIR ) + string( "/" ) + fOutFile;
                }
            }
            else
            {
                if( !fPath.empty() )
                {
                    tFile = string( fPath ) + string( "/" ) + GetName() + string( ".vtp" );
                }
                else
                {
                    tFile = string( OUTPUT_DEFAULT_DIR ) + string( "/" ) + GetName() + string( ".vtp" );
                }
            }

            vismsg( eNormal ) << "vtk track terminator painter <" << GetName() << "> is writing <" << fData->GetNumberOfCells() << "> cells to file <" << tFile << ">" << eom;

            vtkSmartPointer< vtkXMLPolyDataWriter > vWriter = fWindow->GetWriter();
            vWriter->SetFileName( tFile.c_str() );
            vWriter->SetDataModeToBinary();
#ifdef VTK6
            vWriter->SetInputData( fData );
#else
            vWriter->SetInput( fData );
#endif
            vWriter->Write();
        }
        return;
    }

}
