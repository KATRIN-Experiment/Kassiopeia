#include "KSWriteROOT.h"
#include "KSaveSettingsProcessor.hh"
#include "KSWritersMessage.h"
#include "KSComponentGroup.h"

namespace Kassiopeia
{

    const int KSWriteROOT::fBufferSize = 64000;
    const int KSWriteROOT::fSplitLevel = 99;
    const string KSWriteROOT::fLabel = string( "KASSIOPEIA_TREE_DATA" );


    KSWriteROOT::Data::Data( KSComponent* aComponent ) :
            fStructure( NULL ),
            fLabel( "" ),
            fType( "" ),
            fPresence( NULL ),
            fIndex( 0 ),
            fLength( 0 ),
            fData( NULL ),
            fComponents()
    {
        MakeTrees( aComponent );
    }
    KSWriteROOT::Data::~Data()
    {
        delete fStructure;
        delete fPresence;
        delete fData;
    }

    void KSWriteROOT::Data::Start( const unsigned int& anIndex )
    {
        fIndex = anIndex;
        fLength = 0;
        return;
    }
    void KSWriteROOT::Data::Fill()
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
    void KSWriteROOT::Data::Stop()
    {
        fPresence->Fill();
        return;
    }

    void KSWriteROOT::Data::MakeTrees( KSComponent* aComponent )
    {
        string tName = aComponent->GetName();

        string tStructureName = tName + string( "_STRUCTURE" );
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
    void KSWriteROOT::Data::MakeBranches( KSComponent* aComponent )
    {
        wtrmsg_debug( "making branches for object <" << aComponent->GetName() << ">" << eom )

        KSComponentGroup* tComponentGroup = aComponent->As< KSComponentGroup >();
        if( tComponentGroup != NULL )
        {
            wtrmsg_debug( "  object <" << aComponent->GetName() << "> is a group" << eom )
            for( unsigned int tIndex = 0; tIndex < tComponentGroup->ComponentCount(); tIndex++ )
            {
                MakeBranches( tComponentGroup->ComponentAt( tIndex ) );
            }
            return;
        }

        string* tString = aComponent->As< string >();
        if( tString != NULL )
        {
            wtrmsg_debug( "  object <" << aComponent->GetName() << "> is a string" << eom )
            fLabel = aComponent->GetName();
            fType = string( "string" );
            fStructure->Fill();
            fData->Branch( aComponent->GetName().c_str(), tString, fBufferSize, fSplitLevel );
            fComponents.push_back( aComponent );
            return;
        }

        KTwoVector* tTwoVector = aComponent->As< KTwoVector >();
        if( tTwoVector != NULL )
        {
            wtrmsg_debug( "  object <" << aComponent->GetName() << "> is a two_vector" << eom )
            fLabel = aComponent->GetName();
            fType = string( "two_vector" );
            fStructure->Fill();
            fData->Branch( (aComponent->GetName() + string( "_x" )).c_str(), &(tTwoVector->X()), fBufferSize, fSplitLevel );
            fData->Branch( (aComponent->GetName() + string( "_y" )).c_str(), &(tTwoVector->Y()), fBufferSize, fSplitLevel );
            fComponents.push_back( aComponent );
            return;
        }
        KThreeVector* tThreeVector = aComponent->As< KThreeVector >();
        if( tThreeVector != NULL )
        {
            wtrmsg_debug( "  object <" << aComponent->GetName() << "> is a three_vector" << eom )
            fLabel = aComponent->GetName();
            fType = string( "three_vector" );
            fStructure->Fill();
            fData->Branch( (aComponent->GetName() + string( "_x" )).c_str(), &(tThreeVector->X()), fBufferSize, fSplitLevel );
            fData->Branch( (aComponent->GetName() + string( "_y" )).c_str(), &(tThreeVector->Y()), fBufferSize, fSplitLevel );
            fData->Branch( (aComponent->GetName() + string( "_z" )).c_str(), &(tThreeVector->Z()), fBufferSize, fSplitLevel );
            fComponents.push_back( aComponent );
            return;
        }

        bool* tBool = aComponent->As< bool >();
        if( tBool != NULL )
        {
            wtrmsg_debug( "  object <" << aComponent->GetName() << "> is a bool" << eom )
            fLabel = aComponent->GetName();
            fType = string( "bool" );
            fStructure->Fill();
            fData->Branch( aComponent->GetName().c_str(), tBool, fBufferSize, fSplitLevel );
            fComponents.push_back( aComponent );
            return;
        }

        unsigned char* tUChar = aComponent->As< unsigned char >();
        if( tUChar != NULL )
        {
            wtrmsg_debug( "  object <" << aComponent->GetName() << "> is an unsigned_char" << eom )
            fLabel = aComponent->GetName();
            fType = string( "unsigned_char" );
            fStructure->Fill();
            fData->Branch( aComponent->GetName().c_str(), tUChar, fBufferSize, fSplitLevel );
            fComponents.push_back( aComponent );
            return;
        }
        char* tChar = aComponent->As< char >();
        if( tChar != NULL )
        {
            wtrmsg_debug( "  object <" << aComponent->GetName() << "> is a char" << eom )
            fLabel = aComponent->GetName();
            fType = string( "char" );
            fStructure->Fill();
            fData->Branch( aComponent->GetName().c_str(), tChar, fBufferSize, fSplitLevel );
            fComponents.push_back( aComponent );
            return;
        }

        unsigned short* tUShort = aComponent->As< unsigned short >();
        if( tUShort != NULL )
        {
            wtrmsg_debug( "  object <" << aComponent->GetName() << "> is an unsigned_short" << eom )
            fLabel = aComponent->GetName();
            fType = string( "unsigned_short" );
            fStructure->Fill();
            fData->Branch( aComponent->GetName().c_str(), tUShort, fBufferSize, fSplitLevel );
            fComponents.push_back( aComponent );
            return;
        }
        short* tShort = aComponent->As< short >();
        if( tShort != NULL )
        {
            wtrmsg_debug( "  object <" << aComponent->GetName() << "> is a short" << eom )
            fLabel = aComponent->GetName();
            fType = string( "short" );
            fStructure->Fill();
            fData->Branch( aComponent->GetName().c_str(), tShort, fBufferSize, fSplitLevel );
            fComponents.push_back( aComponent );
            return;
        }

        unsigned int* tUInt = aComponent->As< unsigned int >();
        if( tUInt != NULL )
        {
            wtrmsg_debug( "  object <" << aComponent->GetName() << "> is a unsigned_int" << eom )
            fLabel = aComponent->GetName();
            fType = string( "unsigned_int" );
            fStructure->Fill();
            fData->Branch( aComponent->GetName().c_str(), tUInt, fBufferSize, fSplitLevel );
            fComponents.push_back( aComponent );
            return;
        }
        int* tInt = aComponent->As< int >();
        if( tInt != NULL )
        {
            wtrmsg_debug( "  object <" << aComponent->GetName() << "> is an int" << eom )
            fLabel = aComponent->GetName();
            fType = string( "int" );
            fStructure->Fill();
            fData->Branch( aComponent->GetName().c_str(), tInt, fBufferSize, fSplitLevel );
            fComponents.push_back( aComponent );
            return;
        }

        unsigned long* tULong = aComponent->As< unsigned long >();
        if( tULong != NULL )
        {
            wtrmsg_debug( "  object <" << aComponent->GetName() << "> is an unsigned_long" << eom )
            fLabel = aComponent->GetName();
            fType = string( "unsigned_long" );
            fStructure->Fill();
            fData->Branch( aComponent->GetName().c_str(), tULong, fBufferSize, fSplitLevel );
            fComponents.push_back( aComponent );
            return;
        }
        long* tLong = aComponent->As< long >();
        if( tLong != NULL )
        {
            wtrmsg_debug( "  object <" << aComponent->GetName() << "> is a long" << eom )
            fLabel = aComponent->GetName();
            fType = string( "long" );
            fStructure->Fill();
            fData->Branch( aComponent->GetName().c_str(), tLong, fBufferSize, fSplitLevel );
            fComponents.push_back( aComponent );
            return;
        }
        long long* tLongLong = aComponent->As< long long >();
        if( tLongLong != NULL )
        {
            wtrmsg_debug( "  object <" << aComponent->GetName() << "> is a long long" << eom )
            fLabel = aComponent->GetName();
            fType = string( "long long" );
            fStructure->Fill();
            fData->Branch( aComponent->GetName().c_str(), tLongLong, fBufferSize, fSplitLevel );
            fComponents.push_back( aComponent );
            return;
        }

        float* tFloat = aComponent->As< float >();
        if( tFloat != NULL )
        {
            wtrmsg_debug( "  object <" << aComponent->GetName() << "> is a float" << eom )
            fLabel = aComponent->GetName();
            fType = string( "float" );
            fStructure->Fill();
            fData->Branch( aComponent->GetName().c_str(), tFloat, fBufferSize, fSplitLevel );
            fComponents.push_back( aComponent );
            return;
        }
        double* tDouble = aComponent->As< double >();
        {
            wtrmsg_debug( "  object <" << aComponent->GetName() << "> is a double" << eom )
            fLabel = aComponent->GetName();
            fType = string( "double" );
            fStructure->Fill();
            fData->Branch( aComponent->GetName().c_str(), tDouble, fBufferSize, fSplitLevel );
            fComponents.push_back( aComponent );
            return;
        }

        wtrmsg( eError ) << "ROOT writer cannot add object <" << aComponent->GetName() << ">" << eom;

        return;
    }

    KSWriteROOT::KSWriteROOT() :
            fBase( "" ),
            fPath( "" ),
            fStepIteration( 1 ),
            fStepIterationIndex( 0 ),
            fFile( NULL ),
            fRunKeys( NULL ),
            fRunData( NULL ),
            fRunComponents(),
            fActiveRunComponents(),
            fRunIndex( 0 ),
            fRunFirstEventIndex( 0 ),
            fRunLastEventIndex( 0 ),
            fRunFirstTrackIndex( 0 ),
            fRunLastTrackIndex( 0 ),
            fRunFirstStepIndex( 0 ),
            fRunLastStepIndex( 0 ),
            fEventKeys( NULL ),
            fEventData( NULL ),
            fEventComponents(),
            fActiveEventComponents(),
            fEventIndex( 0 ),
            fEventFirstTrackIndex( 0 ),
            fEventLastTrackIndex( 0 ),
            fEventFirstStepIndex( 0 ),
            fEventLastStepIndex( 0 ),
            fTrackKeys( NULL ),
            fTrackData( NULL ),
            fTrackComponents(),
            fActiveTrackComponents(),
            fTrackIndex( 0 ),
            fTrackFirstStepIndex( 0 ),
            fTrackLastStepIndex( 0 ),
            fStepComponent( false ),
            fStepKeys( NULL ),
            fStepData( NULL ),
            fStepComponents(),
            fActiveStepComponents(),
            fStepIndex( 0 )
    {
    }
    KSWriteROOT::KSWriteROOT( const KSWriteROOT& aCopy ) :
            KSComponent(),
            fBase( aCopy.fBase ),
            fPath( aCopy.fPath ),
            fStepIteration( aCopy.fStepIteration ),
            fStepIterationIndex( 0 ),
            fFile( NULL ),
            fRunKeys( NULL ),
            fRunData( NULL ),
            fRunComponents(),
            fActiveRunComponents(),
            fRunIndex( 0 ),
            fRunFirstEventIndex( 0 ),
            fRunLastEventIndex( 0 ),
            fRunFirstTrackIndex( 0 ),
            fRunLastTrackIndex( 0 ),
            fRunFirstStepIndex( 0 ),
            fRunLastStepIndex( 0 ),
            fEventKeys( NULL ),
            fEventData( NULL ),
            fEventComponents(),
            fActiveEventComponents(),
            fEventIndex( 0 ),
            fEventFirstTrackIndex( 0 ),
            fEventLastTrackIndex( 0 ),
            fEventFirstStepIndex( 0 ),
            fEventLastStepIndex( 0 ),
            fTrackKeys( NULL ),
            fTrackData( NULL ),
            fTrackComponents(),
            fActiveTrackComponents(),
            fTrackIndex( 0 ),
            fTrackFirstStepIndex( 0 ),
            fTrackLastStepIndex( 0 ),
            fStepComponent( false ),
            fStepKeys( NULL ),
            fStepData( NULL ),
            fStepComponents(),
            fActiveStepComponents(),
            fStepIndex( 0 )
    {
    }
    KSWriteROOT* KSWriteROOT::Clone() const
    {
        return new KSWriteROOT( *this );
    }
    KSWriteROOT::~KSWriteROOT()
    {
    }

    void KSWriteROOT::ExecuteRun()
    {
        wtrmsg_debug( "ROOT writer <" << fName << "> is filling a run" << eom );

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
        fRunData->Fill();

        fRunIndex++;
        fRunFirstEventIndex = fEventIndex;
        fRunFirstTrackIndex = fTrackIndex;
        fRunFirstStepIndex = fStepIndex;

        return;
    }
    void KSWriteROOT::ExecuteEvent()
    {
        wtrmsg_debug( "ROOT writer <" << fName << "> is filling an event" << eom );

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
        fEventData->Fill();

        fEventIndex++;
        fEventFirstTrackIndex = fTrackIndex;
        fEventFirstStepIndex = fStepIndex;

        return;
    }
    void KSWriteROOT::ExecuteTrack()
    {
        wtrmsg_debug( "ROOT writer <" << fName << "> is filling a track" << eom );

        if ( fStepIndex != 0 )
		{
        	fTrackLastStepIndex = fStepIndex - 1;
		}

        for( ComponentIt tIt = fActiveTrackComponents.begin(); tIt != fActiveTrackComponents.end(); tIt++ )
        {
            tIt->second->Fill();
        }
        fTrackData->Fill();

        fTrackIndex++;
        fTrackFirstStepIndex = fStepIndex;

        return;
    }
    void KSWriteROOT::ExecuteStep()
    {
    	if ( fStepIterationIndex % fStepIteration != 0 )
    	{
            wtrmsg_debug( "ROOT writer <" << fName << "> is skipping a step because of step iteration value <"<<fStepIteration<<">" << eom );
    		fStepIterationIndex++;
    		return;
    	}

        if( fStepComponent == true )
        {
            wtrmsg_debug( "ROOT writer <" << fName << "> is filling a step" << eom );

            for( ComponentIt tIt = fActiveStepComponents.begin(); tIt != fActiveStepComponents.end(); tIt++ )
            {
                tIt->second->Fill();
            }
            fStepData->Fill();
        }

        fStepIndex++;
        fStepIterationIndex++;

        return;
    }

    void KSWriteROOT::AddRunComponent( KSComponent* aComponent )
    {
        ComponentIt tIt = fRunComponents.find( aComponent );
        if( tIt == fRunComponents.end() )
        {
            wtrmsg_debug( "ROOT writer is making a new run output called <" << aComponent->GetName() << ">" << eom );

            fFile->File()->cd();
            fKey = aComponent->GetName();
            fRunKeys->Fill();

            Data* tRunData = new Data( aComponent );
            tIt = fRunComponents.insert( ComponentEntry( aComponent, tRunData ) ).first;
        }

        wtrmsg_debug( "ROOT writer is starting a run output called <" << aComponent->GetName() << ">" << eom );

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
    void KSWriteROOT::DeinitializeComponent()
    {
        wtrmsg_debug( "stopping ROOT writer" << eom );

        if( (fFile != NULL) && (fFile->IsOpen() == true) )
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

            StoreConfig();

            fFile->File()->Write( "", TObject::kOverwrite );

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
                delete tIt->second;
            }

            fFile->Close();

            delete fFile;
        }

        return;
    }

    void KSWriteROOT::StoreConfig(string ConfigDirName)
    {
    	std::vector<string> commands = katrin::KSaveSettingsProcessor::getCommands();
		std::vector<string> values = katrin::KSaveSettingsProcessor::getValues();

		for(size_t treeconfindex=0;treeconfindex<commands.size();treeconfindex++)
		{
			wtrmsg_debug( commands.at(treeconfindex) << ": " << values.at(treeconfindex) << eom);
		}
		if( (fFile != NULL) && (fFile->IsOpen() == true) )
		{
		    fFile->File()->cd();
			gDirectory->mkdir(ConfigDirName.c_str());
			gDirectory->cd(ConfigDirName.c_str());

			for(size_t treeconfindex=0;treeconfindex<commands.size();treeconfindex++)
			{
				if(commands.at(treeconfindex) == "cd")
					gDirectory->cd(values.at(treeconfindex).c_str());
				else if(commands.at(treeconfindex) == "mkdir")
				{
					if(gDirectory->GetDirectory(values.at(treeconfindex).c_str()) == 0)
						gDirectory->mkdir(values.at(treeconfindex).c_str());
				}
				else if(commands.at(treeconfindex) == "tobjstring")
				{
					TObjString XML_element = TObjString(values.at(treeconfindex).c_str());
					XML_element.Write();
				}
				else
				{
					//Unknown format
					wtrmsg (eWarning) << "Unknown stored config tree tag: "  << commands.at(treeconfindex) << eom;
				}
			}
			gDirectory->cd("/");
			fFile->File()->Write();
		}
		else
		{
			wtrmsg (eWarning) << "Can not write config tree to ROOT file - no ROOT file opened." << eom;
		}
    }

    STATICINT sKSWriteROOTDict =
        KSDictionary< KSWriteROOT >::AddCommand( &KSWriteROOT::AddRunComponent, &KSWriteROOT::RemoveRunComponent, "add_run_output", "remove_run_output" ) +
        KSDictionary< KSWriteROOT >::AddCommand( &KSWriteROOT::AddEventComponent, &KSWriteROOT::RemoveEventComponent, "add_event_output", "remove_event_output" ) +
        KSDictionary< KSWriteROOT >::AddCommand( &KSWriteROOT::AddTrackComponent, &KSWriteROOT::RemoveTrackComponent, "add_track_output", "remove_track_output" ) +
        KSDictionary< KSWriteROOT >::AddCommand( &KSWriteROOT::AddStepComponent, &KSWriteROOT::RemoveStepComponent, "add_step_output", "remove_step_output" );

}
