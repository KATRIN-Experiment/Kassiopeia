#include "KSComponent.h"

using namespace std;

namespace Kassiopeia
{

    KSComponent::KSComponent() :
            KSObject(),
            fState( eIdle ),
            fParentComponent( NULL ),
            fChildComponents()
    {
    }
    KSComponent::KSComponent( const KSComponent& aCopy ) :
            KSObject( aCopy ),
            fState( aCopy.fState ),
            fParentComponent( NULL ),
            fChildComponents()
    {
    }
    KSComponent::~KSComponent()
    {
    }

    const KSComponent::StateType& KSComponent::State() const
    {
        return fState;
    }

    void KSComponent::Initialize()
    {
        if( fState == eIdle )
        {
            objctmsg_debug( "component <" << this->GetName() << "> initializing" << eom );
            InitializeComponent();
            fState = eInitialized;

            for( vector< KSComponent* >::iterator tIt = fChildComponents.begin(); tIt != fChildComponents.end(); tIt++ )
            {
                (*tIt)->Initialize();
            }

            return;
        }

        if( fState != eInitialized )
        {
            objctmsg( eError ) << "tried to initialize component <" << this->GetName() << "> from state <" << this->fState << ">" << eom;
        }

        return;
    }
    void KSComponent::Deinitialize()
    {
        if( fState == eInitialized )
        {
            objctmsg_debug( "component <" << this->GetName() << "> deinitializing" << eom );
            DeinitializeComponent();
            fState = eIdle;

            for( vector< KSComponent* >::iterator tIt = fChildComponents.begin(); tIt != fChildComponents.end(); tIt++ )
            {
                (*tIt)->Deinitialize();
            }

            return;
        }

        if( fState != eIdle )
        {
            objctmsg( eError ) << "tried to deinitialize component <" << this->GetName() << "> from state <" << this->fState << ">" << eom;
        }

        return;
    }
    void KSComponent::Activate()
    {
        if( fState == eInitialized )
        {
            objctmsg_debug( "component <" << this->GetName() << "> activating" << eom );
            ActivateComponent();
            fState = eActivated;

            for( vector< KSComponent* >::iterator tIt = fChildComponents.begin(); tIt != fChildComponents.end(); tIt++ )
            {
                (*tIt)->Activate();
            }

            return;
        }

        if( fState != eActivated )
        {
            objctmsg( eError ) << "tried to activate component <" << this->GetName() << "> from state <" << this->fState << ">" << eom;
        }

        return;
    }
    void KSComponent::Deactivate()
    {
        if( fState == eActivated )
        {
            objctmsg_debug( "component <" << this->GetName() << "> deactivating" << eom );
            DeactivateComponent();
            fState = eInitialized;

            for( vector< KSComponent* >::iterator tIt = fChildComponents.begin(); tIt != fChildComponents.end(); tIt++ )
            {
                (*tIt)->Deactivate();
            }

            return;
        }

        if( fState != eInitialized )
        {
            objctmsg( eError ) << "tried to deactivate component <" << this->GetName() << "> from state <" << this->fState << ">" << eom;
        }

        return;
    }
    void KSComponent::PushUpdate()
    {
        if( fState == eActivated )
        {
            objctmsg_debug( "component <" << this->GetName() << "> pushing update" << eom );
            PushUpdateComponent();
            fState = eUpdated;

            for( vector< KSComponent* >::iterator tIt = fChildComponents.begin(); tIt != fChildComponents.end(); tIt++ )
            {
                (*tIt)->PushUpdate();
            }

            return;
        }

        if( fState != eUpdated )
        {
            objctmsg( eError ) << "tried to push update component <" << this->GetName() << "> from state <" << this->fState << ">" << eom;
        }

        for( vector< KSComponent* >::iterator tIt = fChildComponents.begin(); tIt != fChildComponents.end(); tIt++ )
        {
            (*tIt)->PushUpdate();
        }

        return;
    }
    void KSComponent::PushDeupdate()
    {
        if( fState == eUpdated )
        {
            objctmsg_debug( "component <" << this->GetName() << "> pushing deupdate" << eom );
            PushDeupdateComponent();
            fState = eActivated;

            for( vector< KSComponent* >::iterator tIt = fChildComponents.begin(); tIt != fChildComponents.end(); tIt++ )
            {
                (*tIt)->PushDeupdate();
            }

            return;
        }

        if( fState != eActivated )
        {
            objctmsg( eError ) << "tried to push deupdate component <" << this->GetName() << "> from state <" << this->fState << ">" << eom;
        }

        for( vector< KSComponent* >::iterator tIt = fChildComponents.begin(); tIt != fChildComponents.end(); tIt++ )
        {
            (*tIt)->PushDeupdate();
        }

        return;
    }
    void KSComponent::PullUpdate()
    {
        if( fState == eActivated )
        {
            if( fParentComponent != NULL )
            {
                fParentComponent->PullUpdate();
            }

            objctmsg_debug( "component <" << this->GetName() << "> pulling update" << eom );
            PullUpdateComponent();
            fState = eUpdated;

            return;
        }

        if( fState != eUpdated )
        {
            objctmsg( eError ) << "tried to pull update component <" << this->GetName() << "> from state <" << this->fState << ">" << eom;
            return;
        }

        if( fParentComponent != NULL )
        {
            fParentComponent->PullUpdate();
        }

        return;
    }
    void KSComponent::PullDeupdate()
    {
        if( fState == eUpdated )
        {
            if( fParentComponent != NULL )
            {
                fParentComponent->PullDeupdate();
            }

            objctmsg_debug( "component <" << this->GetName() << "> pulling deupdate" << eom );
            PullDeupdateComponent();
            fState = eActivated;

            return;
        }

        if( fState != eActivated )
        {
            objctmsg( eError ) << "tried to pull deupdate component <" << this->GetName() << "> from state <" << this->fState << ">" << eom;
            return;
        }

        if( fParentComponent != NULL )
        {
            fParentComponent->PullDeupdate();
        }

        return;
    }

    void KSComponent::InitializeComponent()
    {
        return;
    }
    void KSComponent::DeinitializeComponent()
    {
        return;
    }
    void KSComponent::ActivateComponent()
    {
        return;
    }
    void KSComponent::DeactivateComponent()
    {
        return;
    }
    void KSComponent::PushUpdateComponent()
    {
        return;
    }
    void KSComponent::PushDeupdateComponent()
    {
        return;
    }
    void KSComponent::PullUpdateComponent()
    {
        return;
    }
    void KSComponent::PullDeupdateComponent()
    {
        return;
    }

    void KSComponent::SetParent( KSComponent* aComponent )
    {
        fParentComponent = aComponent;
        return;
    }
    KSComponent* KSComponent::GetParent() const
    {
        return fParentComponent;
    }

    void KSComponent::AddChild( KSComponent* aChild )
    {
        fChildComponents.push_back( aChild );
        return;
    }
    unsigned int KSComponent::GetChildCount() const
    {
        return fChildComponents.size();
    }
    KSComponent* KSComponent::GetChild( const unsigned int& anIndex ) const
    {
        return fChildComponents.at( anIndex );
    }

}
