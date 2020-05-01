#ifndef KSMUTEX_H_
#define KSMUTEX_H_

#include <pthread.h>

namespace Kassiopeia
{

class KSMutex
{
  public:
    KSMutex();
    virtual ~KSMutex();

    bool Trylock();

    void Lock();
    void Unlock();

  private:
    pthread_mutex_t fMutex;
};

}  // namespace Kassiopeia

#endif
