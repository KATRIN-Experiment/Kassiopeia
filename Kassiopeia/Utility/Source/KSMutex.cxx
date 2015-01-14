#include "KSMutex.h"

namespace Kassiopeia
{

    KSMutex::KSMutex()
    {
        pthread_mutex_init( &fMutex, NULL );
    }
    KSMutex::~KSMutex()
    {
        pthread_mutex_destroy (&fMutex);
    }

    bool KSMutex::Trylock()
    {
        if( pthread_mutex_trylock( &fMutex ) == 0 )
        {
            return true;
        }
        return false;
    }

    void KSMutex::Lock()
    {
        pthread_mutex_lock (&fMutex);
        return;
    }
    void KSMutex::Unlock()
    {
        pthread_mutex_unlock (&fMutex);
        return;
    }

}
