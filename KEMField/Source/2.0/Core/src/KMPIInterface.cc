#include "KMPIInterface.hh"

namespace KEMField
{
  KMPIInterface* KMPIInterface::fMPIInterface = 0;

  KMPIInterface::KMPIInterface()
  {
    fProcess    = -1;
    fNProcesses = -1;
  }

  KMPIInterface::~KMPIInterface()
  {

  }

  void KMPIInterface::Initialize(int* argc, char*** argv)
  {
    /* Let the system do what it needs to start up MPI */
    int initialized = 0;
    MPI_Initialized(&initialized);

    if (!initialized)
      MPI_Init(argc, argv);

    /* Get my process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &fProcess);

    /* Find out how many processes are being used */
    MPI_Comm_size(MPI_COMM_WORLD, &fNProcesses);
  }

  void KMPIInterface::Finalize()
  {
    /* Shut down MPI */
    int finalized = 0;
    MPI_Finalized(&finalized);

    if (!finalized)
      MPI_Finalize();
  }

  /**
   * Interface to accessing KMPIInterface.
   */
  KMPIInterface* KMPIInterface::GetInstance()
  {
    if (fMPIInterface == 0)
      fMPIInterface = new KMPIInterface();
    return fMPIInterface;
  }

  /**
   * Ensures that a process written between BeginSequentialProcess() and
   * EndSequentialProcess() is done one processor at a time.
   */
  void KMPIInterface::BeginSequentialProcess()
  {
    int flag = 1;

    GlobalBarrier();

    if (fProcess>0)
      MPI_Recv(&flag,1,MPI_INT,fProcess-1,50,MPI_COMM_WORLD,&fStatus);
  }

  /**
   * @see BeginSequentialProcess()
   */
  void KMPIInterface::EndSequentialProcess()
  {
    int flag;

    if (fProcess < (fNProcesses-1))
      MPI_Send(&flag,1,MPI_INT,fProcess+1,50,MPI_COMM_WORLD);

    GlobalBarrier();
  }

}
