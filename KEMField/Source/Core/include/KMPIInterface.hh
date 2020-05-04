#ifndef KMPIINTERFACE_DEF
#define KMPIINTERFACE_DEF

#include "mpi.h"

#include <string>
#include <vector>

#define LOCAL_RANK_MPI

namespace KEMField
{

class KMPIInterface
{
  public:
    static KMPIInterface* GetInstance();

    void Initialize(int* argc, char*** argv, bool split_mode = true);
    void Finalize();

    int GetProcess() const
    {
        return fProcess;
    }
    int GetNProcesses() const
    {
        return fNProcesses;
    }
    int GetLocalRank() const
    {
        return fLocalRank;
    }
    std::string GetHostName() const
    {
        return fHostName;
    };

    void BeginSequentialProcess();
    void EndSequentialProcess();

    void GlobalBarrier() const
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    //when called, this function must be encountered by all processes
    //or the program will lock up, treat as a global barrier
    void PrintMessage(std::string msg);

    //routines to be used by programs which split the processes into two
    //groups bases on even/odd local process rank
    bool SplitMode()
    {
        return fSplitMode;
    };
    bool IsSplitValid()
    {
        return fValidSplit;
    };
    bool IsEvenGroupMember()
    {
        return fIsEvenGroupMember;
    };
    int GetNSubGroupProcesses()
    {
        return fNSubGroupProcesses;
    }
    int GetSubGroupRank()
    {
        return fSubGroupRank;
    };
    int GetPartnerProcessID()
    {
        return fPartnerProcessID;
    };

    MPI_Group* GetSubGroup()
    {
        if (fIsEvenGroupMember) {
            return &fEvenGroup;
        }
        else {
            return &fOddGroup;
        };
    }

    MPI_Comm* GetSubGroupCommunicator()
    {
        if (fIsEvenGroupMember) {
            return &fEvenCommunicator;
        }
        else {
            return &fOddCommunicator;
        };
    }

    MPI_Group* GetEvenGroup()
    {
        return &fEvenGroup;
    };
    MPI_Group* GetOddGroup()
    {
        return &fOddGroup;
    };
    MPI_Comm* GetEvenCommunicator()
    {
        return &fEvenCommunicator;
    };
    MPI_Comm* GetOddCommunicator()
    {
        return &fOddCommunicator;
    };


  protected:
    KMPIInterface();
    virtual ~KMPIInterface();

    static KMPIInterface* fMPIInterface;

    int fProcess;
    int fNProcesses;
    int fLocalRank;
    std::string fHostName;
    std::vector<int> fCoHostedProcessIDs;

    //groups and communicators for splitting processes into
    //two sets, based on whether they have even/odd (local) ranks
    bool fSplitMode;
    MPI_Group fEvenGroup;        //even process subgroup
    MPI_Group fOddGroup;         //odd process subgroup
    MPI_Comm fEvenCommunicator;  //comm for even group
    MPI_Comm fOddCommunicator;   //comm for odd group
    bool fValidSplit;            //true if the size of the subgroups is equal
    bool fIsEvenGroupMember;     //true if this process is a member of the even subgroup
    int fSubGroupRank;           //rank of this process in its subgroup
    int fNSubGroupProcesses;     //number of processes in the subgroup this process belongs to
    int fPartnerProcessID;       //global rank of partner process in other subgroup

    void DetermineLocalRank();
    void SetupSubGroups();

    MPI_Status fStatus;
};
}  // namespace KEMField

#endif /* KMPIINTERFACE_DEF */
