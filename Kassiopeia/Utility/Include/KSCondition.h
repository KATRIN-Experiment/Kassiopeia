#ifndef KSCONDITION_H_
#define KSCONDITION_H_

#include <pthread.h>

namespace Kassiopeia
{

class KSCondition
{
  public:
    KSCondition();
    ~KSCondition();

    bool IsWaiting();

    void Wait();
    void Release();

  private:
    bool fState;
    pthread_mutex_t fMutex;
    pthread_cond_t fCondition;
};

}  // namespace Kassiopeia

#endif
