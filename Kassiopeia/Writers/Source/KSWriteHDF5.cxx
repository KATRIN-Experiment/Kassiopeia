#include "KSWriteHDF5.h"
#include "KSWritersMessage.h"
#include "KSComponentGroup.h"

using namespace H5;
using namespace std;

namespace Kassiopeia
{

    //const int KSWriteHDF5::fBufferSize = 64000;
    //const int KSWriteHDF5::fSplitLevel = 99;
    //const string KSWriteHDF5::fLabel = string( "KASSIOPEIA_TREE_DATA" );

    KSWriteHDF5::Data::Data( KSComponent* aComponent ) :
            fStructure( NULL ),
            fLabel( "" ),
            fType( "" ),
            fPresence( NULL ),
            fIndex( 0 ),
            fLength( 0 ),
            fData( NULL ),
            fFile( NULL ),
            fComponents()
    {
        MakeDatasets( aComponent );
    }

    KSWriteHDF5::Data::Data( KSComponent* aComponent, hid_t* aFile ) :
            fStructure( NULL ),
            fLabel( "" ),
            fType( "" ),
            fPresence( NULL ),
            fIndex( 0 ),
            fLength( 0 ),
            fData( NULL ),
            fFile( aFile ),
            fComponents()
    {
        MakeDatasets( aComponent );
    }

    KSWriteHDF5::Data::~Data()
    {
        //H5Sclose(...);
        H5Dclose(fStructure);
        H5Dclose(fPresence);
        H5Dclose(fData);
        H5Gclose(fGroup);

        //delete fStructure;
        //delete fPresence;
        //delete fData;
        //delete fGroup;
    }

    void KSWriteHDF5::Data::Start( const unsigned int& anIndex )
    {
        fIndex = anIndex;
        fLength = 0;
        return;
    }
    void KSWriteHDF5::Data::Fill()
    {
        KSComponent* tComponent;
        vector< KSComponent* >::iterator tIt;

        for( tIt = fComponents.begin(); tIt != fComponents.end(); tIt++ )
        {
            tComponent = (*tIt);
            tComponent->PullUpdate();
        }

        fData->Fill();

        for( tIt = fComponents.begin(); tIt != fComponents.end(); tIt++ )
        {
            tComponent = (*tIt);
            tComponent->PullDeupdate();
        }

        fLength++;
        return;
    }
    void KSWriteHDF5::Data::Stop()
    {
        fPresence->Fill();
        return;
    }

    void KSWriteHDF5::Data::MakeDataSets( KSComponent* aComponent )
    {
        string tName = aComponent->GetName();

        string tStructureName = string("/") + tName + string( "/STRUCTURE" );
        hsize_t tStructureDims[2] = { 100, 100 };
        hid_t tStructureSpace = H5Screate_simple( 2, tStructureDims, NULL );
        fStructure = H5Dcreate2(fFile, tStructureName, H5T_NATIVE_CHAR, tStructureSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        fStructure


	    // Create the data space for the attribute.
	    DataSpace attr_dataspace = DataSpace (1, dims );

	    // Create a dataset attribute.
	Attribute attribute = dataset.createAttribute( ATTR_NAME, PredType::STD_I32BE,
	                                          attr_dataspace);

	// Write the attribute data.
	attribute.write( PredType::NATIVE_INT, attr_data);

        fStructure = new TTree( tStructureName.c_str(), tStructureName.c_str() );
        fStructure->Branch( "LABEL", &fLabel, fBufferSize, fSplitLevel );
        fStructure->Branch( "TYPE", &fType, fBufferSize, fSplitLevel );

        string tPresenceName = tName + string( "_PRESENCE" );
        fPresence = new TTree( tPresenceName.c_str(), tPresenceName.c_str() );
        fPresence->Branch( "INDEX", &fIndex, fBufferSize, fSplitLevel );
        fPresence->Branch( "LENGTH", &fLength, fBufferSize, fSplitLevel );

        string tDataName = tName + string( "_DATA" );
        fData = new TTree( tDataName.c_str(), tDataName.c_str() );

        MakeBranches( aComponent );

        return;
    }




    void KSWriteHDF5::ExecuteRun()
    {
        wtrmsg_debug( "HDF5 writer <" << fName << "> is filling a run" << eom );

        if ( fEventIndex != 0 )
        {
			fRunLastEventIndex = fEventIndex - 1;
        }
        if ( fTrackIndex != 0 )
        {
			fRunLastTrackIndex = fTrackIndex - 1;
        }
        if ( fStepIndex != 0 )
        {
			fRunLastStepIndex = fStepIndex - 1;
        }

        for( ComponentIt tIt = fActiveRunComponents.begin(); tIt != fActiveRunComponents.end(); tIt++ )
        {
            tIt->second->Fill();
        }
        FillRun();

        fRunIndex++;
        fRunFirstEventIndex = fEventIndex;
        fRunFirstTrackIndex = fTrackIndex;
        fRunFirstStepIndex = fStepIndex;

        return;
    }
    void KSWriteHDF5::ExecuteEvent()
    {
        wtrmsg_debug( "HDF5 writer <" << fName << "> is filling an event" << eom );

        if ( fTrackIndex != 0 )
        {
			fEventLastTrackIndex = fTrackIndex - 1;
        }
        if ( fStepIndex != 0 )
        {
			fEventLastStepIndex = fStepIndex - 1;
        }

        for( ComponentIt tIt = fActiveEventComponents.begin(); tIt != fActiveEventComponents.end(); tIt++ )
        {
            tIt->second->Fill();
        }
        FillEvent();

        fEventIndex++;
        fEventFirstTrackIndex = fTrackIndex;
        fEventFirstStepIndex = fStepIndex;

        return;
    }
    void KSWriteHDF5::ExecuteTrack()
    {
        wtrmsg_debug( "HDF5 writer <" << fName << "> is filling a track" << eom );

        if ( fStepIndex != 0 )
		{
        	fTrackLastStepIndex = fStepIndex - 1;
		}

        for( ComponentIt tIt = fActiveTrackComponents.begin(); tIt != fActiveTrackComponents.end(); tIt++ )
        {
            tIt->second->Fill();
        }
        FillTrack();

        fTrackIndex++;
        fTrackFirstStepIndex = fStepIndex;

        return;
    }
    void KSWriteHDF5::ExecuteStep()
    {
    	if ( fStepIterationIndex % fStepIteration != 0 )
    	{
            wtrmsg_debug( "HDF5 writer <" << fName << "> is skipping a step because of step iteration value <"<<fStepIteration<<">" << eom );
    		fStepIterationIndex++;
    		return;
    	}

        if( fStepComponent == true )
        {
            wtrmsg_debug( "HDF5 writer <" << fName << "> is filling a step" << eom );

            for( ComponentIt tIt = fActiveStepComponents.begin(); tIt != fActiveStepComponents.end(); tIt++ )
            {
                tIt->second->Fill();
            }
            FillStep();
        }

        fStepIndex++;
        fStepIterationIndex++;

        return;
    }

    void KSWriteHDF5::AddRunComponent( KSComponent* aComponent )
    {
        ComponentIt tIt = fRunComponents.find( aComponent );
        if( tIt == fRunComponents.end() )
        {
            wtrmsg_debug( "HDF5 writer is making a new run output called <" << aComponent->GetName() << ">" << eom );

            fFile->File()->cd();
            fKey = aComponent->GetName();
            fRunKeys->Fill();

            Data* tRunData = new Data( aComponent );
            tIt = fRunComponents.insert( ComponentEntry( aComponent, tRunData ) ).first;
        }

        wtrmsg_debug( "HDF5 writer is starting a run output called <" << aComponent->GetName() << ">" << eom );

        tIt->second->Start( fRunIndex );
        fActiveRunComponents.insert( *tIt );

        return;
    }
    void KSWriteROOT::RemoveRunComponent( KSComponent* aComponent )
    {
        ComponentIt tIt = fActiveRunComponents.find( aComponent );
        if( tIt == fActiveRunComponents.end() )
        {
            wtrmsg( eError ) << "ROOT writer has no run output called <" << aComponent->GetName() << ">" << eom;
        }

        wtrmsg_debug( "ROOT writer is stopping a run output called <" << aComponent->GetName() << ">" << eom );

        tIt->second->Stop();
        fActiveRunComponents.erase( tIt );

        return;
    }

    void KSWriteROOT::AddEventComponent( KSComponent* aComponent )
    {
        ComponentIt tIt = fEventComponents.find( aComponent );
        if( tIt == fEventComponents.end() )
        {
            wtrmsg_debug( "ROOT writer is making a new event output called <" << aComponent->GetName() << ">" << eom );

            fFile->File()->cd();
            fKey = aComponent->GetName();
            fEventKeys->Fill();

            Data* tEventData = new Data( aComponent );
            tIt = fEventComponents.insert( ComponentEntry( aComponent, tEventData ) ).first;
        }

        wtrmsg_debug( "ROOT writer is starting a event output called <" << aComponent->GetName() << ">" << eom );

        tIt->second->Start( fEventIndex );
        fActiveEventComponents.insert( *tIt );

        return;
    }
    void KSWriteROOT::RemoveEventComponent( KSComponent* aComponent )
    {
        ComponentIt tIt = fActiveEventComponents.find( aComponent );
        if( tIt == fActiveEventComponents.end() )
        {
            wtrmsg( eError ) << "ROOT writer has no event output called <" << aComponent->GetName() << ">" << eom;
        }

        wtrmsg_debug( "ROOT writer is stopping a event output called <" << aComponent->GetName() << ">" << eom );

        tIt->second->Stop();
        fActiveEventComponents.erase( tIt );

        return;
    }

    void KSWriteROOT::AddTrackComponent( KSComponent* aComponent )
    {
        ComponentIt tIt = fTrackComponents.find( aComponent );
        if( tIt == fTrackComponents.end() )
        {
            wtrmsg_debug( "ROOT writer is making a new track output called <" << aComponent->GetName() << ">" << eom );

            fFile->File()->cd();
            fKey = aComponent->GetName();
            fTrackKeys->Fill();

            Data* tTrackData = new Data( aComponent );
            tIt = fTrackComponents.insert( ComponentEntry( aComponent, tTrackData ) ).first;
        }

        wtrmsg_debug( "ROOT writer is starting a track output called <" << aComponent->GetName() << ">" << eom );

        tIt->second->Start( fTrackIndex );
        fActiveTrackComponents.insert( *tIt );

        return;
    }
    void KSWriteROOT::RemoveTrackComponent( KSComponent* aComponent )
    {
        ComponentIt tIt = fActiveTrackComponents.find( aComponent );
        if( tIt == fActiveTrackComponents.end() )
        {
            wtrmsg( eError ) << "ROOT writer has no track output called <" << aComponent->GetName() << ">" << eom;
        }

        wtrmsg_debug( "ROOT writer is stopping a track output called <" << aComponent->GetName() << ">" << eom );

        tIt->second->Stop();
        fActiveTrackComponents.erase( tIt );

        return;
    }

    void KSWriteROOT::AddStepComponent( KSComponent* aComponent )
    {
        if( fStepComponent == false )
        {
            fStepComponent = true;

            const unsigned int tTempStepIndex = fStepIndex;
            for( fStepIndex = 0; fStepIndex < tTempStepIndex; ++fStepIndex )
            {
                fStepData->Fill();
            }
            fStepIndex = tTempStepIndex;
        }

        ComponentIt tIt = fStepComponents.find( aComponent );
        if( tIt == fStepComponents.end() )
        {
            wtrmsg_debug( "ROOT writer is making a new step output called <" << aComponent->GetName() << ">" << eom );

            fFile->File()->cd();
            fKey = aComponent->GetName();
            fStepKeys->Fill();

            Data* tStepData = new Data( aComponent );
            tIt = fStepComponents.insert( ComponentEntry( aComponent, tStepData ) ).first;
        }

        wtrmsg_debug( "ROOT writer is starting a step output called <" << aComponent->GetName() << ">" << eom );

        tIt->second->Start( fStepIndex );
        fActiveStepComponents.insert( *tIt );

        return;
    }
    void KSWriteROOT::RemoveStepComponent( KSComponent* aComponent )
    {
        ComponentIt tIt = fActiveStepComponents.find( aComponent );
        if( tIt == fActiveStepComponents.end() )
        {
            wtrmsg( eError ) << "ROOT writer has no step output called <" << aComponent->GetName() << ">" << eom;
        }

        wtrmsg_debug( "ROOT writer is stopping a step output called <" << aComponent->GetName() << ">" << eom );

        tIt->second->Stop();
        fActiveStepComponents.erase( tIt );

        return;
    }

    void KSWriteROOT::InitializeComponent()
    {
        wtrmsg_debug( "starting ROOT writer" << eom );

        fFile = KRootFile::CreateOutputRootFile( fBase );
        if( !fPath.empty() )
        {
            fFile->AddToPaths( fPath );
        }

        if( fFile->Open( KFile::eWrite ) == true )
        {
            TObjString* tLabel = new TObjString( fLabel.c_str() );
            tLabel->Write( "LABEL", TObject::kOverwrite );
            fFile->File()->cd();
            TTree::SetBranchStyle( 1 );

            fRunKeys = new TTree( "RUN_KEYS", "RUN_KEYS" );
            fRunKeys->Branch( "KEY", &fKey, fBufferSize, fSplitLevel );

            fRunData = new TTree( "RUN_DATA", "RUN_DATA" );
            fRunData->Branch( "RUN_INDEX", &fRunIndex, fBufferSize, fSplitLevel );
            fRunData->Branch( "FIRST_EVENT_INDEX", &fRunFirstEventIndex, fBufferSize, fSplitLevel );
            fRunData->Branch( "LAST_EVENT_INDEX", &fRunLastEventIndex, fBufferSize, fSplitLevel );
            fRunData->Branch( "FIRST_TRACK_INDEX", &fRunFirstTrackIndex, fBufferSize, fSplitLevel );
            fRunData->Branch( "LAST_TRACK_INDEX", &fRunLastTrackIndex, fBufferSize, fSplitLevel );
            fRunData->Branch( "FIRST_STEP_INDEX", &fRunFirstStepIndex, fBufferSize, fSplitLevel );
            fRunData->Branch( "LAST_STEP_INDEX", &fRunLastStepIndex, fBufferSize, fSplitLevel );

            fRunIndex = 0;
            fRunFirstEventIndex = 0;
            fRunLastEventIndex = 0;
            fRunFirstTrackIndex = 0;
            fRunLastTrackIndex = 0;
            fRunFirstStepIndex = 0;
            fRunLastStepIndex = 0;

            fEventKeys = new TTree( "EVENT_KEYS", "EVENT_KEYS" );
            fEventKeys->Branch( "KEY", &fKey, fBufferSize, fSplitLevel );

            fEventData = new TTree( "EVENT_DATA", "EVENT_DATA" );
            fEventData->Branch( "EVENT_INDEX", &fEventIndex, fBufferSize, fSplitLevel );
            fEventData->Branch( "FIRST_TRACK_INDEX", &fEventFirstTrackIndex, fBufferSize, fSplitLevel );
            fEventData->Branch( "LAST_TRACK_INDEX", &fEventLastTrackIndex, fBufferSize, fSplitLevel );
            fEventData->Branch( "FIRST_STEP_INDEX", &fEventFirstStepIndex, fBufferSize, fSplitLevel );
            fEventData->Branch( "LAST_STEP_INDEX", &fEventLastStepIndex, fBufferSize, fSplitLevel );

            fEventIndex = 0;
            fEventFirstTrackIndex = 0;
            fEventLastTrackIndex = 0;
            fEventFirstStepIndex = 0;
            fEventLastStepIndex = 0;

            fTrackKeys = new TTree( "TRACK_KEYS", "TRACK_KEYS" );
            fTrackKeys->Branch( "KEY", &fKey, fBufferSize, fSplitLevel );

            fTrackData = new TTree( "TRACK_DATA", "TRACK_DATA" );
            fTrackData->Branch( "TRACK_INDEX", &fTrackIndex, fBufferSize, fSplitLevel );
            fTrackData->Branch( "FIRST_STEP_INDEX", &fTrackFirstStepIndex, fBufferSize, fSplitLevel );
            fTrackData->Branch( "LAST_STEP_INDEX", &fTrackLastStepIndex, fBufferSize, fSplitLevel );

            fTrackIndex = 0;
            fTrackFirstStepIndex = 0;
            fTrackLastStepIndex = 0;

            fStepKeys = new TTree( "STEP_KEYS", "STEP_KEYS" );
            fStepKeys->Branch( "KEY", &fKey, fBufferSize, fSplitLevel );

            fStepData = new TTree( "STEP_DATA", "STEP_DATA" );
            fStepData->Branch( "STEP_INDEX", &fStepIndex, fBufferSize, fSplitLevel );

            fStepIndex = 0;
        }

        return;
    }
    void KSWriteHDF5::DeinitializeComponent()
    {
        wtrmsg_debug( "stopping HDF5 writer" << eom );

        if( (fFile != NULL) /* && (fFile->IsOpen() == true) */ )
        {
            ComponentIt tIt;

            for( tIt = fActiveRunComponents.begin(); tIt != fActiveRunComponents.end(); tIt++ )
            {
                tIt->second->Stop();
            }

            for( tIt = fActiveEventComponents.begin(); tIt != fActiveEventComponents.end(); tIt++ )
            {
                tIt->second->Stop();
            }

            for( tIt = fActiveTrackComponents.begin(); tIt != fActiveTrackComponents.end(); tIt++ )
            {
                tIt->second->Stop();
            }

            for( tIt = fActiveStepComponents.begin(); tIt != fActiveStepComponents.end(); tIt++ )
            {
                tIt->second->Stop();
            }

            for( tIt = fRunComponents.begin(); tIt != fRunComponents.end(); tIt++ )
            {
                delete tIt->second;
            }

            for( tIt = fEventComponents.begin(); tIt != fEventComponents.end(); tIt++ )
            {
                delete tIt->second;
            }

            for( tIt = fTrackComponents.begin(); tIt != fTrackComponents.end(); tIt++ )
            {
                delete tIt->second;
            }

            for( tIt = fStepComponents.begin(); tIt != fStepComponents.end(); tIt++ )
            {
            }

            H5Fclose(fFile);
            delete fFile;
        }

        return;
    }

}
