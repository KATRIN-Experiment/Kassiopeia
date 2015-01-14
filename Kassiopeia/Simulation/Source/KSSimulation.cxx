#include "KSSimulation.h"

namespace Kassiopeia
{

    KSSimulation::KSSimulation() :
            fSeed( 0 ),
            fRun( 0 ),
            fEvents( 0 )
    {
    }
    KSSimulation::KSSimulation( const KSSimulation& aCopy ) :
            fSeed( aCopy.fSeed ),
            fRun( aCopy.fRun ),
            fEvents( aCopy.fEvents )
    {
    }
    KSSimulation* KSSimulation::Clone() const
    {
        return new KSSimulation( *this );
    }
    KSSimulation::~KSSimulation()
    {
    }

    void KSSimulation::SetSeed( const unsigned int& aSeed )
    {
        fSeed = aSeed;
        return;
    }
    const unsigned int& KSSimulation::GetSeed() const
    {
        return fSeed;
    }

    void KSSimulation::SetRun( const unsigned int& aRun )
    {
        fRun = aRun;
        return;
    }
    const unsigned int& KSSimulation::GetRun() const
    {
        return fRun;
    }

    void KSSimulation::SetEvents( const unsigned int& anEvents )
    {
        fEvents = anEvents;
        return;
    }
    const unsigned int& KSSimulation::GetEvents() const
    {
        return fEvents;
    }

    void KSSimulation::AddCommand( KSCommand* aCommand )
    {
        vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            if( aCommand == (*tCommandIt) )
            {
                return;
            }
        }
        fCommands.push_back( aCommand );
        return;
    }
    void KSSimulation::RemoveCommand( KSCommand* aCommand )
    {
        vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            if( aCommand == (*tCommandIt) )
            {
                fCommands.erase( tCommandIt );
            }
        }
        return;
    }

    void KSSimulation::InitializeComponent()
    {
        vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->GetParent()->Initialize();
            (*tCommandIt)->GetChild()->Initialize();
        }
        return;
    }
    void KSSimulation::DeinitializeComponent()
    {
        vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->GetParent()->Deinitialize();
            (*tCommandIt)->GetChild()->Deinitialize();
        }
        return;
    }
    void KSSimulation::ActivateComponent()
    {
        vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->Activate();
            (*tCommandIt)->GetChild()->Activate();
        }
        return;
    }
    void KSSimulation::DeactivateComponent()
    {
        vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->Deactivate();
            (*tCommandIt)->GetChild()->Deactivate();
        }
        return;
    }

}
