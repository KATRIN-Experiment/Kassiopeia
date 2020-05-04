#include "KSCondition.h"

namespace Kassiopeia
{

KSCondition::KSCondition() : fState(false), fMutex(), fCondition()
{
    pthread_mutex_init(&fMutex, nullptr);
    pthread_cond_init(&fCondition, nullptr);
}
KSCondition::~KSCondition()
{
    pthread_cond_destroy(&fCondition);
    pthread_mutex_destroy(&fMutex);
}

bool KSCondition::IsWaiting()
{
    bool StateCopy;
    pthread_mutex_lock(&fMutex);
    StateCopy = fState;
    pthread_mutex_unlock(&fMutex);
    return StateCopy;
}

void KSCondition::Wait()
{
    pthread_mutex_lock(&fMutex);
    fState = true;
    while (fState == true) {
        pthread_cond_wait(&fCondition, &fMutex);
    }
    pthread_mutex_unlock(&fMutex);
    return;
}
void KSCondition::Release()
{
    pthread_mutex_lock(&fMutex);
    fState = false;
    pthread_cond_signal(&fCondition);
    pthread_mutex_unlock(&fMutex);
    return;
}

}  // namespace Kassiopeia
