#include "KMPIInterface.hh"
#include "KEMCout.hh"

#ifdef LOCAL_RANK_MPI
//used to retrieve the host name
//should be available on on POSIX systems
#include <unistd.h>
#include <sstream>
#endif

#define MSG_TAG 999
#define HOST_DETERMINATION_TAG 998
#define LOCALID_DETERMINATION_TAG 997

namespace KEMField
{
  KMPIInterface* KMPIInterface::fMPIInterface = 0;

  KMPIInterface::KMPIInterface()
  {
    fProcess    = -1;
    fNProcesses = -1;
    fLocalRank = -1;
    fSubGroupRank = -1;
    fNSubGroupProcesses = -1;
    fPartnerProcessID = -1;
    fSplitMode = false;
  }

  KMPIInterface::~KMPIInterface()
  {

  }

  void KMPIInterface::Initialize(int* argc, char*** argv, bool split_mode)
  {
    /* Let the system do what it needs to start up MPI */
    int initialized = 0;
    MPI_Initialized(&initialized);

    if (!initialized)
      MPI_Init(argc, argv);

    /* Get my process fProcess */
    MPI_Comm_rank(MPI_COMM_WORLD, &fProcess);

    /* Find out how many processes are being used */
    MPI_Comm_size(MPI_COMM_WORLD, &fNProcesses);

    //now determine the local rank of this process (indexed from zero) on the local host
    //for example, if processes (0,2,5) are running on host A
    //and processes (1,3,4) are running on host B, then the
    //local rank of process 3 is 1, and the locak rank of process 5 is 2
    DetermineLocalRank();

    fSplitMode = split_mode;



    //construct groups/communicators to evenly split up processes
    SetupSubGroups();

    //cannot make a valid split, so revert to standard mode
    if(!fValidSplit){fSplitMode = false;};

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

    //safely print messages in process order, by passing information to the
    //root process for collection before calling cout
    void
    KMPIInterface::PrintMessage(std::string msg)
    {
        unsigned int n_char = msg.size();

        std::vector<unsigned int> in_msg_sizes;
        std::vector<unsigned int> out_msg_sizes;
        in_msg_sizes.resize(fNProcesses, 0);
        out_msg_sizes.resize(fNProcesses, 0);
        in_msg_sizes[fProcess] = n_char;

        //obtain the message sizes from all of the objects
        MPI_Allreduce( &(in_msg_sizes[0]), &(out_msg_sizes[0]), fNProcesses, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

        //compute the total message size
        unsigned int total_msg_size = 0;
        std::vector<unsigned int> msg_start_indexes;
        msg_start_indexes.resize(fNProcesses);
        for(int i=0; i<fNProcesses; i++)
        {
            total_msg_size += out_msg_sizes[i];
        };

        for(int i=0; i<fNProcesses; i++)
        {
            for(int j=0; j<i; j++)
            {
                msg_start_indexes[i] += out_msg_sizes[j];
            }
        };

        //allocate buffers to reduce all of the messages
        std::vector<char> buf; buf.resize(total_msg_size);

        //fill the appropriate section of the buffer
        for(unsigned int i=0; i<msg.size(); i++)
        {
            buf[msg_start_indexes[fProcess] + i] = msg.at(i);
        }

        MPI_Status status;
        if(fProcess == 0)
        {
            for(int i=1; i < fNProcesses; i++)
            {
                MPI_Recv( &(buf[msg_start_indexes[i]]), out_msg_sizes[i], MPI_CHAR, i, MSG_TAG, MPI_COMM_WORLD, &status);
            }
        }
        else
        {
            MPI_Send( &(buf[ msg_start_indexes[fProcess]]), out_msg_sizes[fProcess], MPI_CHAR, 0, MSG_TAG, MPI_COMM_WORLD);
        }

        if(fProcess == 0)
        {
            //convert to string
            std::stringstream final_output;
            for(unsigned int i=0; i<buf.size(); i++)
            {
                final_output << buf[i];
            }
            std::string full_message = final_output.str();

            //split the messages into separate lines so we can correctly use KEMField::cout
            std::vector<std::string> lines;
            std::string::size_type pos = 0;
            std::string::size_type prev = 0;
            while ((pos = full_message.find('\n', prev)) != std::string::npos)
            {
                lines.push_back(full_message.substr(prev, pos - prev));
                prev = pos + 1;
            }

            if(lines.size() == 0)
            {
                //message has no newline characters, so push the whole message into one line
                lines.push_back(full_message);
            }

            //print message line by line
            for(unsigned int i=0; i<lines.size(); i++)
            {
                KEMField::cout<<lines[i]<<KEMField::endl;
            }
        }
    }


    void KMPIInterface::DetermineLocalRank()
    {
        #ifdef LOCAL_RANK_MPI
        //get the machine's hostname
        char host_name[256];
        int ret_val = gethostname(host_name, 256);
        if( ret_val != 0)
        {
            std::cout<<"Host name error!"<<std::endl;
            KMPIInterface::GetInstance()->Finalize();
            std::exit(1);
        };

        std::stringstream hostname_ss;
        int count = 0;
        do
        {
            hostname_ss << host_name[count];
            count++;
        }
        while( host_name[count] != '\0' && count < 256);
        std::string hostname = hostname_ss.str();
        fHostName = hostname;

        //first we have to collect all of the hostnames that are running a process
        unsigned int n_char = hostname.size();
        std::vector<unsigned int> in_msg_sizes;
        std::vector<unsigned int> out_msg_sizes;
        in_msg_sizes.resize(fNProcesses, 0);
        out_msg_sizes.resize(fNProcesses, 0);
        in_msg_sizes[fProcess] = n_char;

        //obtain the message sizes from all of the processes
        MPI_Allreduce( &(in_msg_sizes[0]), &(out_msg_sizes[0]), fNProcesses, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

        //compute the total message size
        unsigned int total_msg_size = 0;
        std::vector<unsigned int> msg_start_indexes;
        msg_start_indexes.resize(fNProcesses);
        for(int i=0; i<fNProcesses; i++)
        {
            total_msg_size += out_msg_sizes[i];
        };

        for(int i=0; i<fNProcesses; i++)
        {
            for(int j=0; j<i; j++)
            {
                msg_start_indexes[i] += out_msg_sizes[j];
            }
        };

        //allocate buffers to reduce all of the messages
        std::vector<char> buf; buf.resize(total_msg_size);

        //fill the appropriate section of the buffer
        for(unsigned int i=0; i<hostname.size(); i++)
        {
            buf[msg_start_indexes[fProcess] + i] = hostname.at(i);
        }

        //reduce the buffer across all processes
        MPI_Status status;
        if(fProcess == 0)
        {
            for(int i=1; i < fNProcesses; i++)
            {
                MPI_Recv( &(buf[msg_start_indexes[i]]), out_msg_sizes[i], MPI_CHAR, i, HOST_DETERMINATION_TAG , MPI_COMM_WORLD, &status);
            }
        }
        else
        {
            unsigned int root_rank = 0;
            MPI_Send( &(buf[ msg_start_indexes[fProcess]]), out_msg_sizes[fProcess], MPI_CHAR, root_rank, HOST_DETERMINATION_TAG, MPI_COMM_WORLD);
        }

        //now broadcast the complete list of hostnames to all processes
        MPI_Bcast( &(buf[0]), total_msg_size, MPI_CHAR, 0, MPI_COMM_WORLD);

        //now every node has a list of all host names
        //we now need to figure out how many other processes are also running on
        //the same host, and how many devices are available on this host
        //then we can distribute the processes equitably to each device

        std::vector< std::string > hostname_list;
        hostname_list.resize(fNProcesses);
        for(int i=0; i<fNProcesses; i++)
        {
            hostname_list[i] = std::string("");
            for(unsigned int j=0; j<out_msg_sizes[i]; j++)
            {
                hostname_list[i].push_back( buf[msg_start_indexes[i] + j]);
            }
        }

        //collect all the process ids of all the process running on this host
        fCoHostedProcessIDs.clear();
        for(int i=0; i<fNProcesses; i++)
        {
            if(hostname == hostname_list[i])
            {
                fCoHostedProcessIDs.push_back(i);
            }
        }

        //determine the 'local' rank of this process
        for(unsigned int i=0; i<fCoHostedProcessIDs.size(); i++)
        {
            if(fCoHostedProcessIDs[i] == fProcess){fLocalRank = i;};
        }

        #endif
    }


    void
    KMPIInterface::SetupSubGroups()
    {
        #ifdef LOCAL_RANK_MPI
        //we need to retrieve the local rank from each process
        //to make a associative map betweek global-rank and local-rank
        std::vector<int> local_ranks;
        local_ranks.resize(fNProcesses);
        local_ranks[fProcess] = fLocalRank;

        //reduce the buffer across all processes
        MPI_Status status;
        if(fProcess == 0)
        {
            for(int i=1; i < fNProcesses; i++)
            {
                MPI_Recv( &(local_ranks[i]), 1, MPI_INT, i, LOCALID_DETERMINATION_TAG, MPI_COMM_WORLD, &status);
            }
        }
        else
        {
            unsigned int root_rank = 0;
            MPI_Send( &( local_ranks[fProcess] ), 1, MPI_INT, root_rank, LOCALID_DETERMINATION_TAG, MPI_COMM_WORLD);
        }

        //now broadcast the complete list of hostnames to all processes
        MPI_Bcast( &(local_ranks[0]), fNProcesses, MPI_INT, 0, MPI_COMM_WORLD);

        //now every process has a list of the local rank associated with every other process
        //now we can proceed to determine which group they below to
        std::vector<int> even_members;
        std::vector<int> odd_members;
        for(int i=0; i<fNProcesses; i++)
        {
            if(local_ranks[i]%2 == 0){even_members.push_back(i);}
            else{odd_members.push_back(i);}
        }

        fValidSplit = false;
        if(even_members.size() == odd_members.size()){fValidSplit = true;}

        //get the world group
        MPI_Group world;
        MPI_Comm_group(MPI_COMM_WORLD, &world);

        //now we go ahead and construct the groups and communicators
        MPI_Group_incl( world, even_members.size(), &(even_members[0]), &fEvenGroup);
        MPI_Group_incl( world, odd_members.size(), &(odd_members[0]), &fOddGroup);

        MPI_Comm_create(MPI_COMM_WORLD, fEvenGroup, &fEvenCommunicator);
        MPI_Comm_create(MPI_COMM_WORLD, fOddGroup, &fOddCommunicator);

        //now we set things up for this process
        if(fLocalRank%2 == 0)
        {
            fIsEvenGroupMember = true;
            MPI_Comm_rank( fEvenCommunicator, &fSubGroupRank);
            fNSubGroupProcesses = even_members.size();
        }
        else
        {
            fIsEvenGroupMember = false;
            MPI_Comm_rank( fOddCommunicator, &fSubGroupRank);
            fNSubGroupProcesses = odd_members.size();
        }

        //finally, if we have a valid split (equal numbers of even and odd process)
        //we pair up processes so they can exchange data
        if(fValidSplit) //must have even number of processes!
        {
            int status = 0;
            int result = 0;
            if(fCoHostedProcessIDs.size()%2 == 0){status = 1;};
            MPI_Allreduce( &status, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            //to make things faster we first try to pair up processes
            //which share the same node/host
            if(result == fNProcesses)
            {
                //we can because each host has an even number of processes
                if(fIsEvenGroupMember){fPartnerProcessID = fCoHostedProcessIDs[fLocalRank+1];}
                else{fPartnerProcessID = fCoHostedProcessIDs[fLocalRank-1];}
            }
            else
            {
                //this isn't possible so we have to pair up processes across nodes
                if(fIsEvenGroupMember){fPartnerProcessID = fProcess+1;}
                else{fPartnerProcessID = fProcess-1;}
            }
        }
        #endif
    }

}
