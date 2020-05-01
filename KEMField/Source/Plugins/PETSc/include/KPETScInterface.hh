#ifndef KPETSCINTERFACE_DEF
#define KPETSCINTERFACE_DEF

#include <petscsys.h>


namespace KEMField
{
class KPETScInterface
{
  public:
    static KPETScInterface* GetInstance();

    PetscErrorCode Initialize(int* argc, char*** argv);
    PetscErrorCode Finalize();

  protected:
    KPETScInterface();
    virtual ~KPETScInterface();

    static KPETScInterface* fPETScInterface;
};

}  // namespace KEMField

#endif /* KPETSCINTERFACE_DEF */
