#include "KSCommandGroup.h"

namespace Kassiopeia
{

    KSCommandGroup::KSCommandGroup() :
            KSCommand(),
            fCommands()
    {
        Set( this );
    }
    KSCommandGroup::KSCommandGroup( const KSCommandGroup& aCopy ) :
            KSCommand( aCopy ),
            fCommands( aCopy.fCommands )
    {
        Set( this );

        for( CommandIt tIt = fCommands.begin(); tIt != fCommands.end(); tIt++ )
        {
            (*tIt) = (*tIt)->Clone();
        }
    }
    KSCommandGroup::~KSCommandGroup()
    {
        for( CommandIt tIt = fCommands.begin(); tIt != fCommands.end(); tIt++ )
        {
            delete (*tIt);
        }
    }

    KSCommandGroup* KSCommandGroup::Clone() const
    {
        return new KSCommandGroup( *this );
    }

    void KSCommandGroup::AddCommand( KSCommand* aCommand )
    {
        if( aCommand->State() != fState )
        {
            objctmsg( eError ) << "tried to add command <" << aCommand->GetName() << "> in state <" << aCommand->State() << "> to command group <" << this->GetName() << "> in state <" << this->fState << ">" << eom;
            return;
        }
        fCommands.push_back( aCommand );
        return;
    }
    void KSCommandGroup::RemoveCommand( KSCommand* aCommand )
    {
        for( CommandIt tIt = fCommands.begin(); tIt != fCommands.end(); tIt++ )
        {
            if( (*tIt) == aCommand )
            {
                fCommands.erase( tIt );
                return;
            }
        }
        objctmsg( eError ) << "cannot remove command <" << aCommand->GetName() << "> from command group <" << this->GetName() << ">" << eom;
        return;
    }

    KSCommand* KSCommandGroup::CommandAt( unsigned int anIndex )
    {
        return fCommands.at( anIndex );
    }
    const KSCommand* KSCommandGroup::CommandAt( unsigned int anIndex ) const
    {
        return fCommands.at( anIndex );
    }
    unsigned int KSCommandGroup::CommandCount() const
    {
        return fCommands.size();
    }

}
