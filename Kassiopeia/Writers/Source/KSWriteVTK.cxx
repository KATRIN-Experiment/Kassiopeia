#include "KSWriteVTK.h"
#include "KSWritersMessage.h"
#include "KSComponentGroup.h"

#include "KFile.h"

using namespace std;

namespace Kassiopeia
{

    KSWriteVTK::KSWriteVTK() :
        fBase( "" ),
        fPath( "" ),
        fTrackPointFlag( false ),
        fTrackPointComponent( NULL ),
        fTrackPointAction( NULL, NULL ),
        fTrackDataFlag( false ),
        fTrackDataComponent( NULL ),
        fTrackDataActions(),
        fStepPointFlag( false ),
        fStepPointComponent( NULL ),
        fStepPointAction( NULL, NULL ),
        fStepDataFlag( false ),
        fStepDataComponent( NULL ),
        fStepDataActions()
    {
    }
    KSWriteVTK::KSWriteVTK( const KSWriteVTK& aCopy ) :
        KSComponent(),
        fBase( aCopy.fBase ),
        fPath( aCopy.fPath ),
        fTrackPointFlag( false ),
        fTrackPointComponent( NULL ),
        fTrackPointAction( NULL, NULL ),
        fTrackDataFlag( false ),
        fTrackDataComponent( NULL ),
        fTrackDataActions(),
        fStepPointFlag( false ),
        fStepPointComponent( NULL ),
        fStepPointAction( NULL, NULL ),
        fStepDataFlag( false ),
        fStepDataComponent( NULL ),
        fStepDataActions()
    {
    }
    KSWriteVTK* KSWriteVTK::Clone() const
    {
        return new KSWriteVTK( *this );
    }
    KSWriteVTK::~KSWriteVTK()
    {
    }

    void KSWriteVTK::SetBase( const string& aBase )
    {
        fBase = aBase;
        return;
    }
    void KSWriteVTK::SetPath( const string& aPath )
    {
        fPath = aPath;
        return;
    }

    void KSWriteVTK::ExecuteRun()
    {
        wtrmsg_debug( "VTK writer <" << fName << "> is filling a run" << eom );
        return;
    }
    void KSWriteVTK::ExecuteEvent()
    {
        wtrmsg_debug( "VTK writer <" << fName << "> is filling an event" << eom );
        BreakTrack();
        return;
    }
    void KSWriteVTK::ExecuteTrack()
    {
        wtrmsg_debug( "VTK writer <" << fName << "> is filling a track" << eom );
        BreakStep();
        FillTrack();
        return;
    }
    void KSWriteVTK::ExecuteStep()
    {
        wtrmsg_debug( "VTK writer <" << GetName() << "> is filling a step" << eom );
        FillStep();
        return;
    }

    void KSWriteVTK::SetTrackPoint( KSComponent* anComponent )
    {
        if( fTrackPointFlag == false )
        {
            if( fTrackPointComponent == NULL )
            {
                wtrmsg_debug( "VTK writer <" << GetName() << "> is adding a track point object" << eom );
                fTrackPointFlag = true;
                fTrackPointComponent = anComponent;
                AddTrackPoint( anComponent );
                return;
            }
            else
            {
                if( fTrackPointComponent == anComponent )
                {
                    wtrmsg_debug( "VTK writer <" << GetName() << "> is enabling a track point object" << eom );
                    fTrackPointFlag = true;
                    return;
                }
                else
                {
                    wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to set a different track point object" << eom;
                    return;
                }
            }
        }
        else
        {
            wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to set a second track point object" << eom;
            return;
        }
    }
    void KSWriteVTK::ClearTrackPoint( KSComponent* anComponent )
    {
        if( fTrackPointFlag == false )
        {
            wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to clear a cleared track point object" << eom;
            return;
        }
        else
        {
            if( fTrackPointComponent == NULL )
            {
                wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to clear a null track point object" << eom;
                return;
            }
            else
            {
                if( fTrackPointComponent == anComponent )
                {
                    wtrmsg_debug( "VTK writer <" << GetName() << "> is clearing a track point object" << eom );
                    BreakTrack();
                    fTrackPointFlag = false;
                    return;
                }
                else
                {
                    wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to clear a different track point object" << eom;
                    return;
                }
            }
        }
    }

    void KSWriteVTK::SetTrackData( KSComponent* anComponent )
    {
        if( fTrackDataFlag == false )
        {
            if( fTrackDataComponent == NULL )
            {
                wtrmsg_debug( "VTK writer <" << GetName() << "> is adding a track data object" << eom );
                fTrackDataFlag = true;
                fTrackDataComponent = anComponent;
                AddTrackData( anComponent );
                return;
            }
            else
            {
                if( fTrackDataComponent == anComponent )
                {
                    wtrmsg_debug( "VTK writer <" << GetName() << "> is enabling a track data object" << eom );
                    fTrackDataFlag = true;
                    return;
                }
                else
                {
                    wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to set a different track data object" << eom;
                    return;
                }
            }
        }
        else
        {
            wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to set a second track data object" << eom;
            return;
        }
    }
    void KSWriteVTK::ClearTrackData( KSComponent* anComponent )
    {
        if( fTrackDataFlag == false )
        {
            wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to clear a cleared track data object" << eom;
            return;
        }
        else
        {
            if( fTrackDataComponent == NULL )
            {
                wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to clear a null track data object" << eom;
                return;
            }
            else
            {
                if( fTrackDataComponent == anComponent )
                {
                    wtrmsg_debug( "VTK writer <" << GetName() << "> is clearing a track data object" << eom );
                    BreakTrack();
                    fTrackDataFlag = false;
                    return;
                }
                else
                {
                    wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to clear a different track data object" << eom;
                    return;
                }
            }
        }
    }

    void KSWriteVTK::SetStepPoint( KSComponent* anComponent )
    {
        if( fStepPointFlag == false )
        {
            if( fStepPointComponent == NULL )
            {
                wtrmsg_debug( "VTK writer <" << GetName() << "> is adding a step point object" << eom );
                fStepPointFlag = true;
                fStepPointComponent = anComponent;
                AddStepPoint( anComponent );
                return;
            }
            else
            {
                if( fStepPointComponent == anComponent )
                {
                    wtrmsg_debug( "VTK writer <" << GetName() << "> is enabling a step point object" << eom );
                    fStepPointFlag = true;
                    return;
                }
                else
                {
                    wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to set a different step point object" << eom;
                    return;
                }
            }
        }
        else
        {
            wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to set a second step point object" << eom;
            return;
        }
    }
    void KSWriteVTK::ClearStepPoint( KSComponent* anComponent )
    {
        if( fStepPointFlag == false )
        {
            wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to clear a cleared step point object" << eom;
            return;
        }
        else
        {
            if( fStepPointComponent == NULL )
            {
                wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to clear a null step point object" << eom;
                return;
            }
            else
            {
                if( fStepPointComponent == anComponent )
                {
                    wtrmsg_debug( "VTK writer <" << GetName() << "> is clearing a step point object" << eom );
                    BreakStep();
                    fStepPointFlag = false;
                    return;
                }
                else
                {
                    wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to clear a different step point object" << eom;
                    return;
                }
            }
        }
    }

    void KSWriteVTK::SetStepData( KSComponent* anComponent )
    {
        if( fStepDataFlag == false )
        {
            if( fStepDataComponent == NULL )
            {
                wtrmsg_debug( "VTK writer <" << GetName() << "> is adding a step data object" << eom );
                fStepDataFlag = true;
                fStepDataComponent = anComponent;
                AddStepData( anComponent );
                return;
            }
            else
            {
                if( fStepDataComponent == anComponent )
                {
                    wtrmsg_debug( "VTK writer <" << GetName() << "> is enabling a step data object" << eom );
                    fStepDataFlag = true;
                    return;
                }
                else
                {
                    wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to set a different step data object" << eom;
                    return;
                }
            }
        }
        else
        {
            wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to set a second step data object" << eom;
            return;
        }
    }
    void KSWriteVTK::ClearStepData( KSComponent* anComponent )
    {
        if( fStepDataFlag == false )
        {
            wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to clear a cleared step data object" << eom;
            return;
        }
        else
        {
            if( fStepDataComponent == NULL )
            {
                wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to clear a null step data object" << eom;
                return;
            }
            else
            {
                if( fStepDataComponent == anComponent )
                {
                    wtrmsg_debug( "VTK writer <" << GetName() << "> is clearing a step data object" << eom );
                    BreakStep();
                    fStepDataFlag = false;
                    return;
                }
                else
                {
                    wtrmsg( eError ) << "VTK writer <" << GetName() << "> tried to clear a different step data object" << eom;
                    return;
                }
            }
        }
    }

    void KSWriteVTK::InitializeComponent()
    {
        wtrmsg_debug( "starting VTK writer" << eom );

        fStepPoints = vtkSmartPointer< vtkPoints >::New();
        fStepLines = vtkSmartPointer< vtkCellArray >::New();
        fStepData = vtkSmartPointer< vtkPolyData >::New();

        fStepData->SetPoints( fStepPoints );
        fStepData->SetLines( fStepLines );

        fTrackPoints = vtkSmartPointer< vtkPoints >::New();
        fTrackVertices = vtkSmartPointer< vtkCellArray >::New();
        fTrackData = vtkSmartPointer< vtkPolyData >::New();

        fTrackData->SetPoints( fTrackPoints );
        fTrackData->SetVerts( fTrackVertices );

        return;
    }
    void KSWriteVTK::DeinitializeComponent()
    {
        wtrmsg_debug( "stopping VTK writer" << eom );

        if( fBase.length() == 0 )
        {
            fBase = GetName();
        }

        if( fPath.length() == 0 )
        {
            fPath = OUTPUT_DEFAULT_DIR;
        }

        vtkSmartPointer< vtkXMLPolyDataWriter > tStepWriter = vtkSmartPointer< vtkXMLPolyDataWriter >::New();
        tStepWriter->SetFileName( (fPath + string( "/" ) + fBase + string( "Step.vtp" )).c_str() );
        tStepWriter->SetDataModeToBinary();
#ifdef VTK6
        tStepWriter->SetInputData( fStepData );
#else
        tStepWriter->SetInput( fStepData );
#endif
        tStepWriter->Write();

        vtkSmartPointer< vtkXMLPolyDataWriter > tTrackWriter = vtkSmartPointer< vtkXMLPolyDataWriter >::New();
        tTrackWriter->SetFileName( (fPath + string( "/" ) + fBase + string( "Track.vtp" )).c_str() );
        tTrackWriter->SetDataModeToBinary();
#ifdef VTK6
        tTrackWriter->SetInputData( fTrackData );
#else
        tTrackWriter->SetInput( fTrackData );
#endif
        tTrackWriter->Write();

        return;
    }

    void KSWriteVTK::AddTrackPoint( KSComponent* anComponent )
    {
        wtrmsg_debug( "VTK writer <" << GetName() << "> making track point action for object <" << anComponent->GetName() << ">" << eom )

        KThreeVector* tThreeVector = anComponent->As< KThreeVector >();
        if( tThreeVector != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a three_vector" << eom )
            fTrackPointAction.first = anComponent;
            fTrackPointAction.second = new PointAction( tThreeVector, fTrackIds, fTrackPoints );
            return;
        }

        wtrmsg( eError ) << "VTK writer <" << GetName() << "> cannot make point action for object <" << anComponent->GetName() << ">" << eom;

        return;
    }

    void KSWriteVTK::AddTrackData( KSComponent* anComponent )
    {
        wtrmsg_debug( "VTK writer <" << GetName() << "> making track data action for object <" << anComponent->GetName() << ">" << eom )

        KSComponentGroup* tComponentGroup = anComponent->As< KSComponentGroup >();
        if( tComponentGroup != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a group" << eom )
            for( unsigned int tIndex = 0; tIndex < tComponentGroup->ComponentCount(); tIndex++ )
            {
                AddTrackData( tComponentGroup->ComponentAt( tIndex ) );
            }
            return;
        }

        string* tString = anComponent->As< string >();
        if( tString != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a string" << eom )
            wtrmsg( eWarning ) << "  ignoring string object <" << anComponent->GetName() << ">" << eom;
            return;
        }

        KTwoVector* tTwoVector = anComponent->As< KTwoVector >();
        if( tTwoVector != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a two_vector" << eom )
            vtkSmartPointer< vtkDoubleArray > tArray = vtkSmartPointer< vtkDoubleArray >::New();
            tArray->SetName( anComponent->GetName().c_str() );
            tArray->SetNumberOfComponents( 2 );
            fTrackData->GetPointData()->AddArray( tArray );
            fTrackDataActions.insert( ActionEntry( anComponent, new TwoVectorAction( tTwoVector, tArray ) ) );
            return;
        }
        KThreeVector* tThreeVector = anComponent->As< KThreeVector >();
        if( tThreeVector != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a three_vector" << eom )
            vtkSmartPointer< vtkDoubleArray > tArray = vtkSmartPointer< vtkDoubleArray >::New();
            tArray->SetName( anComponent->GetName().c_str() );
            tArray->SetNumberOfComponents( 3 );
            fTrackData->GetPointData()->AddArray( tArray );
            fTrackDataActions.insert( ActionEntry( anComponent, new ThreeVectorAction( tThreeVector, tArray ) ) );
            return;
        }

        bool* tBool = anComponent->As< bool >();
        if( tBool != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a bool" << eom )
            wtrmsg( eWarning ) << "  ignoring bool object <" << anComponent->GetName() << ">" << eom;
            return;
        }

        unsigned short* tUShort = anComponent->As< unsigned short >();
        if( tUShort != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is an unsigned_short" << eom )
            wtrmsg( eWarning ) << "  ignoring unsigned short object <" << anComponent->GetName() << ">" << eom;
            return;
        }
        short* tShort = anComponent->As< short >();
        if( tShort != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a short" << eom )
            wtrmsg( eWarning ) << "  ignoring short object <" << anComponent->GetName() << ">" << eom;
            return;
        }

        unsigned int* tUInt = anComponent->As< unsigned int >();
        if( tUInt != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a unsigned_int" << eom )
            vtkSmartPointer< vtkUnsignedIntArray > tArray = vtkSmartPointer< vtkUnsignedIntArray >::New();
            tArray->SetName( anComponent->GetName().c_str() );
            tArray->SetNumberOfComponents( 1 );
            fTrackData->GetPointData()->AddArray( tArray );
            fTrackDataActions.insert( ActionEntry( anComponent, new UIntAction( tUInt, tArray ) ) );
            return;
        }
        int* tInt = anComponent->As< int >();
        if( tInt != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is an int" << eom )
            vtkSmartPointer< vtkIntArray > tArray = vtkSmartPointer< vtkIntArray >::New();
            tArray->SetName( anComponent->GetName().c_str() );
            tArray->SetNumberOfComponents( 1 );
            fTrackData->GetPointData()->AddArray( tArray );
            fTrackDataActions.insert( ActionEntry( anComponent, new IntAction( tInt, tArray ) ) );
            return;
        }

        unsigned long* tULong = anComponent->As< unsigned long >();
        if( tULong != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is an unsigned_long" << eom )
            wtrmsg( eWarning ) << "  ignoring unsigned long object <" << anComponent->GetName() << ">" << eom;
            return;
        }
        long* tLong = anComponent->As< long >();
        if( tLong != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a long" << eom )
            wtrmsg( eWarning ) << "  ignoring long object <" << anComponent->GetName() << ">" << eom;
            return;
        }

        float* tFloat = anComponent->As< float >();
        if( tFloat != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a float" << eom )
            vtkSmartPointer< vtkFloatArray > tArray = vtkSmartPointer< vtkFloatArray >::New();
            tArray->SetName( anComponent->GetName().c_str() );
            tArray->SetNumberOfComponents( 1 );
            fTrackData->GetPointData()->AddArray( tArray );
            fTrackDataActions.insert( ActionEntry( anComponent, new FloatAction( tFloat, tArray ) ) );
            return;
        }
        double* tDouble = anComponent->As< double >();
        if( tDouble != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a double" << eom )
            vtkSmartPointer< vtkDoubleArray > tArray = vtkSmartPointer< vtkDoubleArray >::New();
            tArray->SetName( anComponent->GetName().c_str() );
            tArray->SetNumberOfComponents( 1 );
            fTrackData->GetPointData()->AddArray( tArray );
            fTrackDataActions.insert( ActionEntry( anComponent, new DoubleAction( tDouble, tArray ) ) );
            return;
        }

        wtrmsg( eError ) << "VTK writer cannot make data action for object <" << anComponent->GetName() << ">" << eom;

        return;
    }

    void KSWriteVTK::FillTrack()
    {
        if( fTrackPointFlag == true )
        {
            if( fTrackDataFlag == true )
            {
                wtrmsg_debug( "VTK writer <" << GetName() << "> is filling a track" << eom );

                fTrackPointAction.first->PullUpdate();
                fTrackPointAction.second->Execute();
                fTrackPointAction.first->PullDeupdate();

                for( ActionIt tIt = fTrackDataActions.begin(); tIt != fTrackDataActions.end(); tIt++ )
                {
                    tIt->first->PullUpdate();
                    tIt->second->Execute();
                    tIt->first->PullDeupdate();
                }
            }
        }

        return;
    }

    void KSWriteVTK::BreakTrack()
    {
        if( fTrackPointFlag == true )
        {
            if( fTrackDataFlag == true )
            {
                if( fTrackIds.empty() == false )
                {
                    wtrmsg_debug( "VTK writer <" << GetName() << "> is breaking a track set with <" << fTrackIds.size() << "> elements" << eom );

                    vtkSmartPointer< vtkVertex > tVertex = vtkSmartPointer< vtkVertex >::New();
                    for( unsigned int tIndex = 0; tIndex < fTrackIds.size(); tIndex++ )
                    {
                        tVertex->GetPointIds()->SetId( tIndex, fTrackIds.at( tIndex ) );
                        fTrackVertices->InsertNextCell( tVertex );
                    }
                    fTrackIds.clear();
                }
            }
        }

        return;
    }

    void KSWriteVTK::AddStepPoint( KSComponent* anComponent )
    {
        wtrmsg_debug( "VTK writer <" << GetName() << "> making step point action for object <" << anComponent->GetName() << ">" << eom )

        KThreeVector* tThreeVector = anComponent->As< KThreeVector >();
        if( tThreeVector != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a three_vector" << eom )
            fStepPointAction.first = anComponent;
            fStepPointAction.second = new PointAction( tThreeVector, fStepIds, fStepPoints );
            return;
        }

        wtrmsg( eError ) << "VTK writer <" << GetName() << "> cannot make point action for object <" << anComponent->GetName() << ">" << eom;

        return;
    }

    void KSWriteVTK::AddStepData( KSComponent* anComponent )
    {
        wtrmsg_debug( "VTK writer <" << GetName() << "> making step data action for object <" << anComponent->GetName() << ">" << eom )

        KSComponentGroup* tComponentGroup = anComponent->As< KSComponentGroup >();
        if( tComponentGroup != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a group" << eom )
            for( unsigned int tIndex = 0; tIndex < tComponentGroup->ComponentCount(); tIndex++ )
            {
                AddStepData( tComponentGroup->ComponentAt( tIndex ) );
            }
            return;
        }

        string* tString = anComponent->As< string >();
        if( tString != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a string" << eom )
            wtrmsg( eWarning ) << "  ignoring string object <" << anComponent->GetName() << ">" << eom;
            return;
        }

        KTwoVector* tTwoVector = anComponent->As< KTwoVector >();
        if( tTwoVector != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a two_vector" << eom )
            vtkSmartPointer< vtkDoubleArray > tArray = vtkSmartPointer< vtkDoubleArray >::New();
            tArray->SetName( anComponent->GetName().c_str() );
            tArray->SetNumberOfComponents( 2 );
            fStepData->GetPointData()->AddArray( tArray );
            fStepDataActions.insert( ActionEntry( anComponent, new TwoVectorAction( tTwoVector, tArray ) ) );
            return;
        }
        KThreeVector* tThreeVector = anComponent->As< KThreeVector >();
        if( tThreeVector != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a three_vector" << eom )
            vtkSmartPointer< vtkDoubleArray > tArray = vtkSmartPointer< vtkDoubleArray >::New();
            tArray->SetName( anComponent->GetName().c_str() );
            tArray->SetNumberOfComponents( 3 );
            fStepData->GetPointData()->AddArray( tArray );
            fStepDataActions.insert( ActionEntry( anComponent, new ThreeVectorAction( tThreeVector, tArray ) ) );
            return;
        }

        bool* tBool = anComponent->As< bool >();
        if( tBool != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a bool" << eom )
            wtrmsg( eWarning ) << "  ignoring bool object <" << anComponent->GetName() << ">" << eom;
            return;
        }

        unsigned short* tUShort = anComponent->As< unsigned short >();
        if( tUShort != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is an unsigned_short" << eom )
            wtrmsg( eWarning ) << "  ignoring unsigned short object <" << anComponent->GetName() << ">" << eom;
            return;
        }
        short* tShort = anComponent->As< short >();
        if( tShort != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a short" << eom )
            wtrmsg( eWarning ) << "  ignoring short object <" << anComponent->GetName() << ">" << eom;
            return;
        }

        unsigned int* tUInt = anComponent->As< unsigned int >();
        if( tUInt != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a unsigned_int" << eom )
            vtkSmartPointer< vtkUnsignedIntArray > tArray = vtkSmartPointer< vtkUnsignedIntArray >::New();
            tArray->SetName( anComponent->GetName().c_str() );
            tArray->SetNumberOfComponents( 1 );
            fStepData->GetPointData()->AddArray( tArray );
            fStepDataActions.insert( ActionEntry( anComponent, new UIntAction( tUInt, tArray ) ) );
            return;
        }
        int* tInt = anComponent->As< int >();
        if( tInt != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is an int" << eom )
            vtkSmartPointer< vtkIntArray > tArray = vtkSmartPointer< vtkIntArray >::New();
            tArray->SetName( anComponent->GetName().c_str() );
            tArray->SetNumberOfComponents( 1 );
            fStepData->GetPointData()->AddArray( tArray );
            fStepDataActions.insert( ActionEntry( anComponent, new IntAction( tInt, tArray ) ) );
            return;
        }

        unsigned long* tULong = anComponent->As< unsigned long >();
        if( tULong != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is an unsigned_long" << eom )
            wtrmsg( eWarning ) << "  ignoring unsigned long object <" << anComponent->GetName() << ">" << eom;
            return;
        }
        long* tLong = anComponent->As< long >();
        if( tLong != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a long" << eom )
            wtrmsg( eWarning ) << "  ignoring long object <" << anComponent->GetName() << ">" << eom;
            return;
        }

        float* tFloat = anComponent->As< float >();
        if( tFloat != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a float" << eom )
            vtkSmartPointer< vtkFloatArray > tArray = vtkSmartPointer< vtkFloatArray >::New();
            tArray->SetName( anComponent->GetName().c_str() );
            tArray->SetNumberOfComponents( 1 );
            fStepData->GetPointData()->AddArray( tArray );
            fStepDataActions.insert( ActionEntry( anComponent, new FloatAction( tFloat, tArray ) ) );
            return;
        }
        double* tDouble = anComponent->As< double >();
        if( tDouble != NULL )
        {
            wtrmsg_debug( "  object <" << anComponent->GetName() << "> is a double" << eom )
            vtkSmartPointer< vtkDoubleArray > tArray = vtkSmartPointer< vtkDoubleArray >::New();
            tArray->SetName( anComponent->GetName().c_str() );
            tArray->SetNumberOfComponents( 1 );
            fStepData->GetPointData()->AddArray( tArray );
            fStepDataActions.insert( ActionEntry( anComponent, new DoubleAction( tDouble, tArray ) ) );
            return;
        }

        wtrmsg( eError ) << "VTK writer cannot make data action for object <" << anComponent->GetName() << ">" << eom;

        return;
    }

    void KSWriteVTK::FillStep()
    {
        if( fStepPointFlag == true )
        {
            if( fStepDataFlag == true )
            {
                wtrmsg_debug( "VTK writer <" << GetName() << "> is filling a step" << eom );

                fStepPointAction.first->PullUpdate();
                fStepPointAction.second->Execute();
                fStepPointAction.first->PullDeupdate();

                for( ActionIt tIt = fStepDataActions.begin(); tIt != fStepDataActions.end(); tIt++ )
                {
                    tIt->first->PullUpdate();
                    tIt->second->Execute();
                    tIt->first->PullDeupdate();
                }
            }
        }

        return;
    }

    void KSWriteVTK::BreakStep()
    {
        if( fStepPointFlag == true )
        {
            if( fStepDataFlag == true )
            {
                if( fStepIds.empty() == false )
                {
                    wtrmsg_debug( "VTK writer <" << GetName() << "> is breaking a step set with <" << fStepIds.size() << "> elements" << eom );

                    vtkSmartPointer< vtkPolyLine > tPolyLine = vtkSmartPointer< vtkPolyLine >::New();
                    tPolyLine->GetPointIds()->SetNumberOfIds( fStepIds.size() );
                    for( unsigned int tIndex = 0; tIndex < fStepIds.size(); tIndex++ )
                    {
                        tPolyLine->GetPointIds()->SetId( tIndex, fStepIds.at( tIndex ) );
                    }
                    fStepIds.clear();
                    fStepLines->InsertNextCell( tPolyLine );
                }
            }
        }

        return;
    }

    STATICINT sKSWriteVTKDict =
        KSDictionary< KSWriteVTK >::AddCommand( &KSWriteVTK::SetStepPoint, &KSWriteVTK::ClearStepPoint, "set_step_point", "clear_step_point" ) +
        KSDictionary< KSWriteVTK >::AddCommand( &KSWriteVTK::SetStepData, &KSWriteVTK::ClearStepData, "set_step_data", "clear_step_data" ) +
        KSDictionary< KSWriteVTK >::AddCommand( &KSWriteVTK::SetTrackPoint, &KSWriteVTK::ClearTrackPoint, "set_track_point", "clear_track_point" ) +
        KSDictionary< KSWriteVTK >::AddCommand( &KSWriteVTK::SetTrackData, &KSWriteVTK::ClearTrackData, "set_track_data", "clear_track_data" );

}
