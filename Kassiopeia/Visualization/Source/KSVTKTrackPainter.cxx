#include "KSVTKTrackPainter.h"
#include "KSVisualizationMessage.h"

#include "KSReadFileROOT.h"
#include "KRootFile.h"

#include "vtkPointData.h"
#include "vtkPolyLine.h"

using namespace std;

namespace Kassiopeia
{
    KSVTKTrackPainter::KSVTKTrackPainter() :
            fPoints( vtkSmartPointer< vtkPoints >::New() ),
            fLines( vtkSmartPointer< vtkCellArray >::New() ),
            fColors( vtkSmartPointer< vtkDoubleArray >::New() ),
            fData( vtkSmartPointer< vtkPolyData >::New() ),
            fMapper( vtkSmartPointer< vtkPolyDataMapper >::New() ),
            fTable( vtkSmartPointer< vtkLookupTable >::New() ),
            fActor( vtkSmartPointer< vtkActor >::New() )
    {
        fColors->SetNumberOfComponents( 1 );
        fData->SetPoints( fPoints );
        fData->SetLines( fLines );
        fData->GetPointData()->SetScalars( fColors );
        fTable->SetNumberOfTableValues( 256 );
        fTable->SetHueRange( 0.000, 0.667 );
        fTable->SetRampToLinear();
        fTable->SetVectorModeToMagnitude();
        fTable->Build();
#ifdef VTK6
        fMapper->SetInputData( fData );
#else
        fMapper->SetInput( fData );
#endif
        fMapper->SetScalarModeToUsePointData();
        fMapper->SetLookupTable( fTable );
        fMapper->SetColorModeToMapScalars();
        fMapper->ScalarVisibilityOn();
        fActor->SetMapper( fMapper );
    }
    KSVTKTrackPainter::~KSVTKTrackPainter()
    {
    }

    void KSVTKTrackPainter::Render()
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
        KSReadStepROOT& tStep = tReader.GetStep();

        KSReadObjectROOT& tPointObject = tStep.GetObject( fPointObject );
        const KSThreeVector& tPointVariable = tPointObject.Get< KSThreeVector >( fPointVariable );

        KSReadObjectROOT& tColorObject = tStep.GetObject( fColorObject );
        const KSDouble& tColorVariable = tColorObject.Get< KSDouble >( fColorVariable );

        bool tActive;
        vector< vtkIdType > tIds;

        for( unsigned int tRunIndex = 0; tRunIndex <= tRun.GetLastRunIndex(); tRunIndex++ )
        {
            tRun << tRunIndex;
            for( unsigned int tEventIndex = tRun.GetFirstEventIndex(); tEventIndex <= tRun.GetLastEventIndex(); tEventIndex++ )
            {
                tEvent << tEventIndex;
                for( long tTrackIndex = tEvent.GetFirstTrackIndex(); tTrackIndex <= tEvent.GetLastTrackIndex(); tTrackIndex++ )
                {
                    tTrack << tTrackIndex;

                    tActive = false;

                    for( long tStepIndex = tTrack.GetFirstStepIndex(); tStepIndex <= tTrack.GetLastStepIndex(); tStepIndex++ )
                    {
                        tStep << tStepIndex;

                        if( tActive == false )
                        {
                            if( (tPointObject.Valid() == true) && (tColorObject.Valid() == true) )
                            {
                                vismsg_debug( "output became active at <" << tStepIndex << ">" << eom );

                                fColors->InsertNextValue( tColorVariable.Value() );
                                tIds.push_back( fPoints->InsertNextPoint( tPointVariable.Value().X(), tPointVariable.Value().Y(), tPointVariable.Value().Z() ) );

                                tActive = true;
                            }
                        }
                        else
                        {
                            if( (tPointObject.Valid() == true) && (tColorObject.Valid() == true) )
                            {
                                fColors->InsertNextValue( tColorVariable.Value() );
                                tIds.push_back( fPoints->InsertNextPoint( tPointVariable.Value().X(), tPointVariable.Value().Y(), tPointVariable.Value().Z() ) );
                            }
                            else
                            {
                                vismsg_debug( "output became inactive at <" << tStepIndex << ">" << eom );

                                if( tIds.empty() == false )
                                {
                                    vismsg_debug( "making polyline of <" << tIds.size() << "> points" << eom );

                                    for( unsigned int tIndex = 0; tIndex < tIds.size(); tIndex++ )
                                    {
                                        vtkSmartPointer< vtkPolyLine > tPolyLine = vtkSmartPointer< vtkPolyLine >::New();
                                        tPolyLine->GetPointIds()->SetNumberOfIds( tIds.size() );
                                        for( unsigned int tIndex = 0; tIndex < tIds.size(); tIndex++ )
                                        {
                                            tPolyLine->GetPointIds()->SetId( tIndex, tIds.at( tIndex ) );
                                        }
                                        tIds.clear();
                                        fLines->InsertNextCell( tPolyLine );
                                    }
                                }

                                tActive = false;
                            }
                        }

                    }

                    if( tIds.empty() == false )
                    {
                        vismsg_debug( "making polyline of <" << tIds.size() << "> points" << eom );

                        for( unsigned int tIndex = 0; tIndex < tIds.size(); tIndex++ )
                        {
                            vtkSmartPointer< vtkPolyLine > tPolyLine = vtkSmartPointer< vtkPolyLine >::New();
                            tPolyLine->GetPointIds()->SetNumberOfIds( tIds.size() );
                            for( unsigned int tIndex = 0; tIndex < tIds.size(); tIndex++ )
                            {
                                tPolyLine->GetPointIds()->SetId( tIndex, tIds.at( tIndex ) );
                            }
                            tIds.clear();
                            fLines->InsertNextCell( tPolyLine );
                        }
                    }
                }
            }
        }

        tReader.CloseFile();

        delete tRootFile;

        fMapper->SetScalarRange( fColors->GetRange() );
        fMapper->Update();

        return;
    }

    void KSVTKTrackPainter::Display()
    {
        if( fDisplayEnabled == true )
        {
            vtkSmartPointer< vtkRenderer > vRenderer = fWindow->GetRenderer();
            vRenderer->AddActor( fActor );
        }

        return;
    }

    void KSVTKTrackPainter::Write()
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

            vismsg( eNormal ) << "vtk track painter <" << GetName() << "> is writing <" << fData->GetNumberOfCells() << "> cells to file <" << tFile << ">" << eom;

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
