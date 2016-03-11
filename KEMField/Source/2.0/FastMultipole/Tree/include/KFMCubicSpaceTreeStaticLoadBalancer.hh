#ifndef __KFMCubicSpaceTreeStaticLoadBalancer_H__
#define __KFMCubicSpaceTreeStaticLoadBalancer_H__

#include "KFMMessaging.hh"
#include "KFMArrayMath.hh"
#include "KMPIInterface.hh"

#include <cstdlib>
#include <cmath>
#include <vector>
#include <list>
#include <algorithm>

#ifdef KEMFIELD_USE_GSL
#include <gsl/gsl_rng.h>
#endif

namespace KEMField
{

/**
*
*@file KFMCubicSpaceTreeStaticLoadBalancer.hh
*@class KFMCubicSpaceTreeStaticLoadBalancer
*@brief Tries to determine the best way to divide work among separate processes
*by trying to make contiguous spatial regions with roughly equal amount of work
*(given by the scores indicated) out of the top level nodes in a cubic tree,
*it assumes that the nodes indices are given by a row-major organization in
*the spatial dimensions
*
*
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jan 16 13:28:55 EST 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NDIM>
struct KFMWorkBlock
{
    unsigned int index;
    unsigned int z_order_index;
    double spatial_coordinates[NDIM];
    double score;
};

template<unsigned int NDIM>
bool KFMWorkBlockCompareMortonZOrder(KFMWorkBlock<NDIM> a, KFMWorkBlock<NDIM> b)
{
    return (a.z_order_index < b.z_order_index);
};

template<unsigned int NDIM>
class KFMCubicSpaceTreeStaticLoadBalancer
{
    public:

        KFMCubicSpaceTreeStaticLoadBalancer();
        virtual ~KFMCubicSpaceTreeStaticLoadBalancer();

        void SetVerbosity(int verbosity){fVerbosity = verbosity;};
        void SetMaxIterations(unsigned int max_iter){fMaxIterations = max_iter;};

        void SetBlockLength(double len);
        void SetDivisions(unsigned int div);
        void SetNeighborOrder(unsigned int order);
        void SetAlpha(double a){fAlpha = a;};
        void SetBeta(double b){fBeta = b;};
        void SetGamma(double c){fGamma = c;};
        //set the id's of the blocks with available work
        void SetBlocks(const std::vector< KFMWorkBlock<NDIM> >* work_blocks);

        //must be called before estimating work balance
        void Initialize();

        //estimate the distribution of blocks over process to balance the load
        virtual void EstimateBalancedWork();

        //get the list of blocks assigned to this process
        void GetProcessBlockIdentities(std::vector<unsigned int>* process_block_ids) const;

    protected:

        //determines the best state found among the mpi processes
        void DetermineBestAvailableState();

        //functions needed for simulated annealing
        void PermuteState(); //generates a new state from the current one
        void UpdateState(); //updates current state from the permuted one
        double UniformRandom(double lower_limit, double upper_limit); //uses rand to generate double on [0,1]
        double ComputePermutedStateEnergyFunction(); //computes max energy over all process states (permuted state)
        double ComputeProcessEnergyFunction(const std::list< KFMWorkBlock<NDIM> >* proc_blocks); //computes energy of single process state
        double ComputeTemperatureFunction(int iteration);
        double BlockCenterDistance(const double* p1, const double* p2);

        /* data */
        unsigned int fNProcesses;//number of processes
        unsigned int fProcessID; //the id of this process
        unsigned int fVerbosity;
        unsigned int fMaxIterations;
        unsigned int fDivisions;
        unsigned int fNMaxBlocks;
        unsigned int fDimensions[NDIM];
        unsigned int fNeighborOrder;
        double fBlockLength;

        //ids and scores of each block with work
        //it is assumed that these id's are assigned according
        //to row-major spatial ordering of all blocks
        //and the scores are proportional to the amount of work in each block
        std::vector< KFMWorkBlock<NDIM> > fAllBlocks;
        unsigned int fNBlocks;
        double fMeanBlockScore;
        double fBlockScoreVariance;
        unsigned int fMaxWorkitems;

        //the proxy identies (references to the id/score pair)
        //of the blocks that are assigned to each process in
        //order to balance the workload
        std::vector< std::list< KFMWorkBlock<NDIM> > > fProcessBlockSets; //current good state
        std::vector< std::list< KFMWorkBlock<NDIM> > > fPermutedProcessBlockSets; //permuted state for next step

        //solution, list of block ids with corresponding list of processes ids
        std::vector<unsigned int> fSolutionBlockIDs;
        std::vector<unsigned int> fSolutionProcessIDs;

        //variables for annealing
        double fAlpha; //weight factor for raw score
        double fBeta; //weight factor for region continuity
        double fGamma; //temperature reduction factor (0<fGamma<1)
        double fInitialTemperature;

        double fCurrentStateEnergy;
        double fPermutedStateEnergy;

        //keep memory of the best state found so far
        double fGlobalMinimumStateEnergy;
        std::vector< std::list< KFMWorkBlock<NDIM> > > fGlobalMinimumBlockSets;

        #ifdef KEMFIELD_USE_GSL
        gsl_rng* fR;
        #endif
};


template<unsigned int NDIM>
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::KFMCubicSpaceTreeStaticLoadBalancer()
{
    if( KMPIInterface::GetInstance()->SplitMode() )
    {
        //the load balance should only be run by one sub-group (processes with even local id's)
        fNProcesses = KMPIInterface::GetInstance()->GetNSubGroupProcesses();
        fProcessID = KMPIInterface::GetInstance()->GetSubGroupRank();
    }
    else
    {
        fNProcesses = KMPIInterface::GetInstance()->GetNProcesses();
        fProcessID = KMPIInterface::GetInstance()->GetProcess();
    }

    fDivisions = 0;
    fNMaxBlocks = 0;
    for(unsigned int i=0; i<NDIM; i++){fDimensions[i] = 0;};
    fNeighborOrder = 0;
    fBlockLength = 0.0;

    fAllBlocks.clear();
    fNBlocks = 0;
    fMeanBlockScore = 0.0;
    fBlockScoreVariance = 0.0;
    fMaxWorkitems = 0;
    fVerbosity = 0;
    fMaxIterations = 1;

    fProcessBlockSets.clear();
    fPermutedProcessBlockSets.clear();

    fAlpha = 1.0;
    fBeta = 0.0;
    fGamma = 0.95;
    fInitialTemperature = 0.0;
    fCurrentStateEnergy = 0.0;
    fPermutedStateEnergy = 0.0;

    #ifdef KEMFIELD_USE_GSL
    const gsl_rng_type* T;
    gsl_rng_env_setup();
    T = gsl_rng_default; //default is mt199937
    fR = gsl_rng_alloc(T);
    #endif
}

template<unsigned int NDIM>
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::~KFMCubicSpaceTreeStaticLoadBalancer()
{
    #ifdef KEMFIELD_USE_GSL
    gsl_rng_free(fR);
    #endif
};


template<unsigned int NDIM>
void
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::SetBlockLength(double len)
{
    fBlockLength = len;
};

template<unsigned int NDIM>
void
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::SetDivisions(unsigned int div)
{
    fDivisions = div;
    fNMaxBlocks = 1;
    for(unsigned int i=0; i<NDIM; i++)
    {
        fDimensions[i] = fDivisions;
        fNMaxBlocks *= fDivisions;
    };
}

template<unsigned int NDIM>
void
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::SetNeighborOrder(unsigned int order)
{
    fNeighborOrder = order;
}

template<unsigned int NDIM>
void
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::SetBlocks(const std::vector< KFMWorkBlock<NDIM> >* work_blocks)
{
    fAllBlocks = *work_blocks;
    fNBlocks = fAllBlocks.size();
}

template<unsigned int NDIM>
void
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::Initialize()
{
    //calculate normalization for all of the block scores
    //we normalize w.r.t to the maximum score
    double max_score = 0.0;
    for(unsigned int i=0; i<fNBlocks; i++)
    {
        if(fAllBlocks[i].score > max_score){max_score = fAllBlocks[i].score;};
    }

    //compute the morton z-indices of all of the blocks
    //also compute the mean score
    fMeanBlockScore = 0.0;
    for(unsigned int i=0; i<fNBlocks; i++)
    {
        //normaliz block score
        fAllBlocks[i].score /= max_score;
        //note that morton z-ordering works best for spatial sorting
        //when using power-of-two dimension sizes
        unsigned int offset = fAllBlocks[i].index;
        unsigned int z_index = KFMArrayMath::MortonZOrderFromOffset<NDIM>(offset, fDimensions);
        fAllBlocks[i].z_order_index = z_index;
        fMeanBlockScore += fAllBlocks[i].score;
    }
    fMeanBlockScore /= (double)fNBlocks;

    //now compute the block score variance
    fBlockScoreVariance = 0.0;
    for(unsigned int i=0; i<fNBlocks; i++)
    {
        double del = fAllBlocks[i].score - fMeanBlockScore;
        fBlockScoreVariance += del*del;
    }
    fBlockScoreVariance = std::sqrt(fBlockScoreVariance)/( (double)fNBlocks );

    //now sort the blocks according to the morton z-order
    std::sort(fAllBlocks.begin(), fAllBlocks.end(), KFMWorkBlockCompareMortonZOrder<NDIM>);

    fSolutionBlockIDs.resize(fNBlocks);
    fSolutionProcessIDs.resize(fNBlocks);

    if(fVerbosity >= 3)
    {
        if(fProcessID == 0)
        {
            kfmout<<"KFMCubicSpaceTreeStaticLoadBalancer::Initialize: Block score mean: "<<fMeanBlockScore<<kfmendl;
            kfmout<<"KFMCubicSpaceTreeStaticLoadBalancer::Initialize: Block score variance: "<<fBlockScoreVariance<<kfmendl;
        }
    }

}


template<unsigned int NDIM>
void
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::EstimateBalancedWork()
{
    //we determine which processes should handle which blocks by stochastically
    //minimizing a energy/cost function subject to the rule that each process
    //must recieve at least one block.
    if(fNProcesses < fNBlocks)
    {
        //make a naive estimate of how many blocks to allocate to each process
        //this estimate assumes each block has equal weight/score
        std::vector<unsigned int> process_n_workitems;
        process_n_workitems.resize(fNProcesses, 0);
        fMaxWorkitems = 1;
        for(unsigned int i=0; i<fNBlocks; i++)
        {
            process_n_workitems[i%fNProcesses] += 1;
            if(process_n_workitems[i%fNProcesses] > fMaxWorkitems){fMaxWorkitems = process_n_workitems[i%fNProcesses];};
        }

        //now make naive association of blocks to processes
        //work blocks are ordered on morton z
        fProcessBlockSets.clear();
        fPermutedProcessBlockSets.clear();
        fProcessBlockSets.resize(fNProcesses);
        fPermutedProcessBlockSets.resize(fNProcesses);
        fGlobalMinimumBlockSets.resize(fNProcesses);

        unsigned int count = 0;
        for(unsigned int i=0; i<fNProcesses; i++)
        {
            fProcessBlockSets[i].clear();
            fPermutedProcessBlockSets[i].clear();
            fGlobalMinimumBlockSets[i].clear();
            for(unsigned int j=0; j<process_n_workitems[i]; j++)
            {
                fProcessBlockSets[i].push_back( fAllBlocks[count] );
                fPermutedProcessBlockSets[i].push_back( fAllBlocks[count] );
                fGlobalMinimumBlockSets[i].push_back( fAllBlocks[count] );
                count++;
            }
        }

        fCurrentStateEnergy = ComputePermutedStateEnergyFunction();
        fGlobalMinimumStateEnergy = fCurrentStateEnergy;
        fInitialTemperature = 0.5*fCurrentStateEnergy;

        if(fVerbosity >= 3)
        {
            if(fProcessID == 0)
            {
                //send message about the initial scores of each process
                //their mean and their variance
                std::vector<double> proc_score(fNProcesses, 0.0);
                for(unsigned int i=0; i<fNProcesses; i++)
                {
                    proc_score[i] = ComputeProcessEnergyFunction( &(fPermutedProcessBlockSets[i]) );
                }


                double max_proc_score = 0.0;
                double min_proc_score = proc_score[0];
                double ave_proc_score = 0.0;
                for(unsigned int i=0; i<fNProcesses; i++)
                {
                    if(max_proc_score < proc_score[i]){max_proc_score = proc_score[i];};
                    if(min_proc_score > proc_score[i]){min_proc_score = proc_score[i];};
                    ave_proc_score += proc_score[i];
                }
                ave_proc_score /= (double)fNProcesses;

                double var_proc_score = 0.0;
                for(unsigned int i=0; i<fNProcesses; i++)
                {
                    double del = proc_score[i] - ave_proc_score;
                    var_proc_score += del*del;
                }
                var_proc_score = (1.0/((double)fNProcesses) )*std::sqrt(var_proc_score);

                kfmout<<"KFMCubicSpaceTreeStaticLoadBalancer::EstimateBalancedWork: Initial max process score: "<<max_proc_score<<kfmendl;
                kfmout<<"KFMCubicSpaceTreeStaticLoadBalancer::EstimateBalancedWork: Initial min process score: "<<min_proc_score<<kfmendl;
                kfmout<<"KFMCubicSpaceTreeStaticLoadBalancer::EstimateBalancedWork: Initial process score mean: "<<ave_proc_score<<kfmendl;
                kfmout<<"KFMCubicSpaceTreeStaticLoadBalancer::EstimateBalancedWork: Initial process score variance: "<<var_proc_score<<kfmendl;
            }
        }

        //since optimal load balancing is NP-complete we do few iterations
        //of simulated annealing to find a good-enough solution
        bool isFinished = false;
        if(fMaxWorkitems == 1){isFinished = true;};
        if(fNProcesses == 1){isFinished = true;};
        unsigned int iter = 0;

        //seed rand with process id
        #ifndef KEMFIELD_USE_GSL
            srand(fProcessID);
        #else
            gsl_rng_set(fR, fProcessID);
        #endif

        while(!isFinished)
        {
            PermuteState();
            fPermutedStateEnergy = ComputePermutedStateEnergyFunction();

            //if we found a lower energy state, we transition to it
            if(fPermutedStateEnergy < fCurrentStateEnergy)
            {
                fCurrentStateEnergy = fPermutedStateEnergy;
                UpdateState();
            }
            else
            {
                //we have some probability of transitioning to a state
                //with slightly worse energy
                double t = ComputeTemperatureFunction(iter);
                double probability = std::exp( -1.0*( (fPermutedStateEnergy - fCurrentStateEnergy)/t) );
                double dice = UniformRandom(0.0, 1.0);
                if(dice < probability)
                {
                    fCurrentStateEnergy = fPermutedStateEnergy;
                    UpdateState();
                }
            }

            //put a hard limit on the amount minimization work we do
            iter++;
            if(iter >= fMaxIterations){isFinished = true;};
        };

        //since we randomized on process id, the MPI threads results
        //have diverged, now have to determine the best process distribution
        //found by any of processes
        DetermineBestAvailableState();
    }
    else
    {
        //error, abort, too many processes allocated for this job
        kfmout<<"KFMCubicSpaceTreeStaticLoadBalancer::EstimateBalancedWork(): Error, too many MPI processes allocated for this job. ";
        kfmout<<"The total number work blocks is: "<<fNBlocks<<" but the number of processes is: "<<fNProcesses<<". ";
        kfmout<<"Please either increase the granularity of the spatial divisions or allocate fewer processes for this job. "<<kfmendl;
        #ifdef KEMFIELD_USE_MPI
        KMPIInterface::GetInstance()->Finalize();
        #endif
        kfmexit(1);
    }

}


//functions needed for simulated annealing
template<unsigned int NDIM>
void
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::PermuteState()
{
    //set the permuted state equal to the current state
    for(unsigned int i=0; i<fNProcesses; i++)
    {
        fPermutedProcessBlockSets[i] = fProcessBlockSets[i];
    }

    //select the giving/recieving processes
    unsigned int give_process = 0;
    unsigned int recieve_process = 0;
    unsigned int count = 0;
    bool found = true;
    bool finished = false;
    do
    {
        found = true;
        finished = true;

        give_process = UniformRandom(0, fNProcesses);
        recieve_process = UniformRandom(0, fNProcesses);

        if(fPermutedProcessBlockSets[give_process].size() < 2)
        {
            found = false;
            finished = false;
        }

        if(give_process == recieve_process)
        {
            found = false;
            finished = false;
        }

        count++;

        //if we have tried more than 100*fNProcesses times, don't keep trying
        // to find a combination
        if(count >= 100*fNProcesses){finished = true;};
    }
    while(!finished);

    //select a random block from the giving process
    if(found)
    {
        unsigned int r = static_cast<unsigned int>( UniformRandom(0.0, fPermutedProcessBlockSets[give_process].size() ) );
        typename std::list< KFMWorkBlock<NDIM> >::iterator it = fPermutedProcessBlockSets[give_process].begin();
        if(r ==  fPermutedProcessBlockSets[give_process].size() ){r--;};
        for(unsigned int i=0; i<r; i++){it++;};

        //transfer the block
        if(it != fPermutedProcessBlockSets[give_process].end() )
        {
            fPermutedProcessBlockSets[recieve_process].push_back(*it);
            fPermutedProcessBlockSets[give_process].erase(it);
        }
    }
}

template<unsigned int NDIM>
void
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::UpdateState()
{
    for(unsigned int i=0; i<fNProcesses; i++)
    {
        fProcessBlockSets[i] = fPermutedProcessBlockSets[i];
    }

    if(fCurrentStateEnergy < fGlobalMinimumStateEnergy)
    {
        fGlobalMinimumStateEnergy = fCurrentStateEnergy;
        for(unsigned int i=0; i<fNProcesses; i++)
        {
            fGlobalMinimumBlockSets[i] = fProcessBlockSets[i];
        }
    }
}


template<unsigned int NDIM>
double
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::UniformRandom(double lower_limit, double upper_limit)
{
    double r = 0;
    #ifndef KEMFIELD_USE_GSL
        //we don't need high quality random numbers here, so we use rand()
        double m = RAND_MAX;
        m += 1;// do not want the range to be inclusive of the upper limit
        double r1 = rand();
        r = r1/m;
    #else
        //gsl is available, so use it instead
        r = gsl_rng_uniform(fR);
    #endif
    return lower_limit + (upper_limit - lower_limit)*r;
}

template<unsigned int NDIM>
double
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::ComputePermutedStateEnergyFunction()
{
    double max_proc_score = 0.0;
    for(unsigned int i=0; i<fNProcesses; i++)
    {
        double proc_score = ComputeProcessEnergyFunction( &(fPermutedProcessBlockSets[i]) );
        if(proc_score > max_proc_score){max_proc_score = proc_score;};
    }
    return max_proc_score;
}

template<unsigned int NDIM>
double
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::ComputeProcessEnergyFunction(const std::list< KFMWorkBlock<NDIM> >* proc_blocks)
{
    //compute contributions due to raw score of each block
    double score_energy = 0.0;

    typename std::list< KFMWorkBlock<NDIM> >::const_iterator it;
    for(it = proc_blocks->begin(); it != proc_blocks->end(); it++)
    {
        score_energy += (*it).score;
    }

    //compute contribution to energy due spatial proximity of the
    //blocks, this introduces a penalty for non-contiguous regions
    //this term is proportional to the surface/volume ratio of process block set
    //we assume a unit length of 1 for the sides of blocks
    double prox_energy = 0.0;
    double face_area = std::pow(fBlockLength,NDIM-1);

    //may want to change the surface area calculation to avoid counting faces
    //which are adjacent to empty regions (blocks w/ no work)
    double surface_area = 2.0*NDIM*face_area*proc_blocks->size();
    double volume = std::pow(fBlockLength, NDIM)*proc_blocks->size();
    double max_ratio = surface_area/volume;
    //now perform double loop over blocks to subtract off surface
    //area of the sides which are adjacent
    typename std::list< KFMWorkBlock<NDIM> >::const_iterator it2;
    for(it = proc_blocks->begin(); it != proc_blocks->end(); it++)
    {
        for(it2 = proc_blocks->begin(); it2 != proc_blocks->end(); it2++)
        {
            double dist = BlockCenterDistance( &( ((*it).spatial_coordinates)[0] ) ,  &( ((*it2).spatial_coordinates)[0] )  );
            if(dist < std::sqrt(NDIM)*fBlockLength)
            {
                //blocks are adjacent so subtrace off the area of one face
                surface_area -= face_area;
            }
        }
    }

    //compute normalized surface/volume ratio
    prox_energy = (surface_area/volume)/max_ratio;

    //return weighted score as energy
    return fAlpha*score_energy + fBeta*prox_energy;
}


template<unsigned int NDIM>
double
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::ComputeTemperatureFunction(int iteration)
{
    return fInitialTemperature*std::pow(fGamma, iteration);
}



template<unsigned int NDIM>
void
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::DetermineBestAvailableState()
{
    unsigned int best_process_id;
    if(fMaxWorkitems == 1)
    {
        //system has no degrees of freedom to balance work just take the initial
        //distribution of blocks from process 0
        best_process_id = 0;
    }
    else
    {
        //determine which process has the state with the lowest energy
        std::vector<double> energies_in(fNProcesses, 0.0);
        std::vector<double> energies_out(fNProcesses, 0.0);
        energies_in[fProcessID] = fGlobalMinimumStateEnergy;

        if(KMPIInterface::GetInstance()->SplitMode())
        {
            MPI_Comm* subgroup_comm = KMPIInterface::GetInstance()->GetSubGroupCommunicator();
            MPI_Allreduce( &(energies_in[0]), &(energies_out[0]), fNProcesses, MPI_DOUBLE, MPI_SUM, *subgroup_comm);
        }
        else
        {
            MPI_Allreduce( &(energies_in[0]), &(energies_out[0]), fNProcesses, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }

        double min_energy = energies_out[0];
        best_process_id = 0;
        for(unsigned int i=0; i<fNProcesses; i++)
        {
            if(energies_out[i] < min_energy){min_energy = energies_out[i]; best_process_id = i;};
        }
    }

    unsigned int count = 0;
    for(unsigned int i=0; i<fNProcesses; i++)
    {
        typename std::list< KFMWorkBlock<NDIM> >::const_iterator it;
        for(it = fGlobalMinimumBlockSets[i].begin(); it != fGlobalMinimumBlockSets[i].end(); it++)
        {
            fSolutionProcessIDs[count] = i;
            fSolutionBlockIDs[count] = (*it).index;
            count++;
        }
    }


    if(fVerbosity >= 3)
    {
        if(fProcessID == best_process_id)
        {
            //send message about the initial scores of each process
            //their mean and their variance
            std::vector<double> proc_score(fNProcesses, 0.0);
            for(unsigned int i=0; i<fNProcesses; i++)
            {
                proc_score[i] = ComputeProcessEnergyFunction( &(fGlobalMinimumBlockSets[i]) );
            }

            double max_proc_score = 0.0;
            double min_proc_score = proc_score[0];
            double ave_proc_score = 0.0;
            for(unsigned int i=0; i<fNProcesses; i++)
            {
                if(max_proc_score < proc_score[i]){max_proc_score = proc_score[i];};
                if(min_proc_score > proc_score[i]){min_proc_score = proc_score[i];};
                ave_proc_score += proc_score[i];
            }
            ave_proc_score /= (double)fNProcesses;

            double var_proc_score = 0.0;
            for(unsigned int i=0; i<fNProcesses; i++)
            {
                double del = proc_score[i] - ave_proc_score;
                var_proc_score += del*del;
            }
            var_proc_score = (1.0/((double)fNProcesses) )*std::sqrt(var_proc_score);

            kfmout<<"KFMCubicSpaceTreeStaticLoadBalancer::DetermineBestAvailableState: Final max process score: "<<max_proc_score<<kfmendl;
            kfmout<<"KFMCubicSpaceTreeStaticLoadBalancer::DetermineBestAvailableState: Final min process score: "<<min_proc_score<<kfmendl;
            kfmout<<"KFMCubicSpaceTreeStaticLoadBalancer::DetermineBestAvailableState: Final process score mean: "<<ave_proc_score<<kfmendl;
            kfmout<<"KFMCubicSpaceTreeStaticLoadBalancer::DetermineBestAvailableState: Final process score variance: "<<var_proc_score<<kfmendl;
        }
    }

    //now broadcast the best solution information to the other nodes
    if(KMPIInterface::GetInstance()->SplitMode())
    {
        MPI_Comm* subgroup_comm = KMPIInterface::GetInstance()->GetSubGroupCommunicator();
        MPI_Bcast( &(fSolutionBlockIDs[0]), fNBlocks, MPI_UNSIGNED, best_process_id, *subgroup_comm);
        MPI_Bcast( &(fSolutionProcessIDs[0]), fNBlocks, MPI_UNSIGNED, best_process_id, *subgroup_comm);
    }
    else
    {
        MPI_Bcast( &(fSolutionBlockIDs[0]), fNBlocks, MPI_UNSIGNED, best_process_id, MPI_COMM_WORLD);
        MPI_Bcast( &(fSolutionProcessIDs[0]), fNBlocks, MPI_UNSIGNED, best_process_id, MPI_COMM_WORLD);
    }

}

template<unsigned int NDIM>
void
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::GetProcessBlockIdentities(std::vector<unsigned int>* process_block_ids) const
{
    std::vector<unsigned int> blocks;
    for(unsigned int i=0; i<fNBlocks; i++)
    {
        if(fSolutionProcessIDs[i] == fProcessID)
        {
            blocks.push_back(fSolutionBlockIDs[i]);
        }
    }
    *process_block_ids = blocks;
}

template<unsigned int NDIM>
double
KFMCubicSpaceTreeStaticLoadBalancer<NDIM>::BlockCenterDistance( const double* p1, const double* p2)
{
    double val = 0.0;
    for(unsigned int i=0; i<NDIM; i++)
    {
        double del = p1[i] - p2[i];
        val += del*del;
    }
    return std::sqrt(val);
}




}

#endif /* __KFMCubicSpaceTreeStaticLoadBalancer_H__ */
