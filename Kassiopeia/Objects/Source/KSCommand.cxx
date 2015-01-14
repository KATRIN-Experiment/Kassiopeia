#include "KSCommand.h"

namespace Kassiopeia
{

    KSCommand::KSCommand() :
            KSObject(),
            fState( eIdle ),
            fParentComponent( NULL ),
            fChildComponent( NULL )
    {
    }
    KSCommand::KSCommand( const KSCommand& aCopy ) :
            KSObject( aCopy ),
            fState( aCopy.fState ),
            fParentComponent( aCopy.fParentComponent ),
            fChildComponent( aCopy.fChildComponent )
    {
    }
    KSCommand::~KSCommand()
    {
    }

    const KSCommand::StateType& KSCommand::State() const
    {
        return fState;
    }

    void KSCommand::Activate()
    {
        if( fState == eIdle )
        {
            objctmsg_debug( "command <" << this->GetName() << "> activating" << eom );
            ActivateCommand();
            fState = eActivated;

            return;
        }

        if( fState != eActivated )
        {
            objctmsg( eError ) << "tried to activate command <" << this->GetName() << "> from state <" << this->fState << ">" << eom;
        }

        return;
    }
    void KSCommand::Deactivate()
    {
        if( fState == eActivated )
        {
            objctmsg_debug( "command <" << this->GetName() << "> deactivating" << eom );
            DeactivateCommand();
            fState = eIdle;

            return;
        }

        if( fState != eIdle )
        {
            objctmsg( eError ) << "tried to deactivate command <" << this->GetName() << "> from state <" << this->fState << ">" << eom;
        }

        return;
    }

    void KSCommand::ActivateCommand()
    {
        return;
    }
    void KSCommand::DeactivateCommand()
    {
        return;
    }

    void KSCommand::SetParent( KSComponent* aComponent )
    {
        fParentComponent = aComponent;
        return;
    }
    KSComponent* KSCommand::GetParent() const
    {
        return fParentComponent;
    }

    void KSCommand::SetChild( KSComponent* aComponent )
    {
        fChildComponent = aComponent;
        return;
    }
    KSComponent* KSCommand::GetChild() const
    {
        return fChildComponent;
    }

}
