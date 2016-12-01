#include "KSSimulation.h"

using namespace std;

namespace Kassiopeia
{

    KSSimulation::KSSimulation() :
            fSeed( 0 ),
            fRun( 0 ),
            fEvents( 0 ),
            fStepReportIteration( 1000 ),
            fCommands()
    {
    }
    KSSimulation::KSSimulation( const KSSimulation& aCopy ) :
            KSComponent(),
            fSeed( aCopy.fSeed ),
            fRun( aCopy.fRun ),
            fEvents( aCopy.fEvents ),
            fStepReportIteration( aCopy.fStepReportIteration ),
            fCommands()
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

    void KSSimulation::SetStepReportIteration( const unsigned int& anIteration )
    {
        fStepReportIteration = anIteration;
        return;
    }
    const unsigned int& KSSimulation::GetStepReportIteration() const
    {
        return fStepReportIteration;
    }

    void KSSimulation::AddCommand( KSCommand* aCommand )
    {
        std::vector< KSCommand* >::iterator tCommandIt;
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
        std::vector< KSCommand* >::iterator tCommandIt;
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
        std::vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->GetParent()->Initialize();
            (*tCommandIt)->GetChild()->Initialize();
        }
        return;
    }
    void KSSimulation::DeinitializeComponent()
    {
        std::vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->GetParent()->Deinitialize();
            (*tCommandIt)->GetChild()->Deinitialize();
        }
        return;
    }
    void KSSimulation::ActivateComponent()
    {
        std::vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->Activate();
            (*tCommandIt)->GetChild()->Activate();
        }
        return;
    }
    void KSSimulation::DeactivateComponent()
    {
        std::vector< KSCommand* >::iterator tCommandIt;
        for( tCommandIt = fCommands.begin(); tCommandIt != fCommands.end(); tCommandIt++ )
        {
            (*tCommandIt)->Deactivate();
            (*tCommandIt)->GetChild()->Deactivate();
        }
        return;
    }

}
