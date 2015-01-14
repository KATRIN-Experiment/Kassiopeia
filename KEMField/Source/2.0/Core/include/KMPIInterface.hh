#ifndef KMPIINTERFACE_DEF
#define KMPIINTERFACE_DEF



#include "mpi.h"

namespace KEMField{

  class KMPIInterface
  {
  public:
    static KMPIInterface* GetInstance();

    void Initialize(int* argc, char*** argv);
    void Finalize();

    int GetProcess()     const { return fProcess; }
    int GetNProcesses()  const { return fNProcesses; }

    void BeginSequentialProcess();
    void EndSequentialProcess();

    void GlobalBarrier() const { MPI_Barrier(MPI_COMM_WORLD); }

  protected:
    KMPIInterface();
    virtual ~KMPIInterface();

    static KMPIInterface* fMPIInterface;

    int fProcess;
    int fNProcesses;
    MPI_Status fStatus;
  };
}

#endif /* KMPIINTERFACE_DEF */
