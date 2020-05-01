/*
 * KElectricFastMultipoleFieldSolver.cc
 *
 *  Created on: 23.07.2015
 *      Author: gosda
 */

#include "KElectricFastMultipoleFieldSolver.hh"

#include "KEMCout.hh"
#include "KEMFileInterface.hh"
#include "KFMElectrostaticFieldMapper_SingleThread.hh"
#include "KFMElectrostaticTreeConstructor.hh"
#include "KFMElectrostaticTreeData.hh"
#include "KMD5HashGenerator.hh"

#include <string>

#ifdef KEMFIELD_USE_OPENCL
#include "KFMElectrostaticFieldMapper_OpenCL.hh"
#include "KOpenCLSurfaceContainer.hh"
#endif

#include "KMPIEnvironment.hh"

using std::string;

namespace KEMField
{

KElectricFastMultipoleFieldSolver::KElectricFastMultipoleFieldSolver() :
    fFastMultipoleFieldSolver(nullptr),
#ifdef KEMFIELD_USE_OPENCL
    fFastMultipoleFieldSolverOpenCL(NULL),
#endif
    fUseOpenCL(false)
{
    fTree = new KFMElectrostaticTree();
}

KElectricFastMultipoleFieldSolver::~KElectricFastMultipoleFieldSolver()
{
    delete fTree;
    delete fFastMultipoleFieldSolver;
#ifdef KEMFIELD_USE_OPENCL
    delete fFastMultipoleFieldSolverOpenCL;

    if (fUseOpenCL) {
        KOpenCLSurfaceContainer* oclContainer =
            dynamic_cast<KOpenCLSurfaceContainer*>(KOpenCLInterface::GetInstance()->GetActiveData());
        if (oclContainer)
            delete oclContainer;
        oclContainer = NULL;
        KOpenCLInterface::GetInstance()->SetActiveData(oclContainer);
    }

#endif
}

void KElectricFastMultipoleFieldSolver::InitializeCore(KSurfaceContainer& container)
{
    //the tree constuctor definitions
    typedef KFMElectrostaticTreeConstructor<KFMElectrostaticFieldMapper_SingleThread> TreeConstructor_SingleThread;
#ifdef KEMFIELD_USE_OPENCL
    typedef KFMElectrostaticTreeConstructor<KFMElectrostaticFieldMapper_OpenCL> TreeConstructor_OpenCL;
    KOpenCLData* data = KOpenCLInterface::GetInstance()->GetActiveData();
    KOpenCLSurfaceContainer* oclContainer;
    if (data) {
        KEMField::cout << "using a prexisting OpenCL surface container." << KEMField::endl;
        oclContainer = dynamic_cast<KOpenCLSurfaceContainer*>(data);
    }
    else {
        KEMField::cout << "creating a new OpenCL surface container." << KEMField::endl;
        oclContainer = new KOpenCLSurfaceContainer(container);
        KOpenCLInterface::GetInstance()->SetActiveData(oclContainer);
    }
#else
    typedef KFMElectrostaticTreeConstructor<KFMElectrostaticFieldMapper_SingleThread> TreeConstructor_OpenCL;
#endif

    // compute hash of the solved geometry
    KMD5HashGenerator solutionHashGenerator;
    string solutionHash = solutionHashGenerator.GenerateHash(container);

    //KEMField::cout << "<shape+boundary+solution> hash is <" << solutionHash << ">" << KEMField::endl;

    // compute hash of the parameter values on the bare geometry
    // compute hash of the parameter values
    KMD5HashGenerator parameterHashGenerator;
    string parameterHash = parameterHashGenerator.GenerateHash(fParameters);

    //KEMField::cout << "<parameter> hash is <" << parameterHash << ">" KEMField::endl;

    // create label set for multipole tree container object
    string fmContainerBase(KFMElectrostaticTreeData::Name());
    string fmContainerName = fmContainerBase + string("_") + solutionHash + string("_") + parameterHash;

    if (fUseOpenCL) {
        fmContainerName += string("_OpenCL");
    }

    auto* tree_data = new KFMElectrostaticTreeData();

    bool containerFound = false;
    KEMFileInterface::GetInstance()->FindByName(*tree_data, fmContainerName, containerFound);

    if (containerFound == true) {
        KEMField::cout << "fast multipole tree found." << KEMField::endl;

        //construct tree from data
        TreeConstructor_SingleThread constructor;
        constructor.ConstructTree(*tree_data, *fTree);
    }
    else {
        //KEMField::cout << "no fast multipole tree found." << KEMField::endl;

        //must construct the tree
        //assign tree parameters and id
        fTree->SetParameters(fParameters);
        fTree->GetTreeProperties()->SetTreeID(fmContainerName);

        //construct the tree
        if (fUseOpenCL) {
            TreeConstructor_OpenCL constructor;
#ifdef KEMFIELD_USE_OPENCL
            constructor.ConstructTree(*oclContainer, *fTree);
#else
            constructor.ConstructTree(container, *fTree);
#endif
        }
        else {
            TreeConstructor_SingleThread constructor;
            constructor.ConstructTree(container, *fTree);
        }

        TreeConstructor_SingleThread constructor;
        constructor.SaveTree(*fTree, *tree_data);

        MPI_SINGLE_PROCESS
        {
            KEMFileInterface::GetInstance()->Write(*tree_data, fmContainerName);
        }
    }

    //now build the field solver
    if (fUseOpenCL) {
#ifdef KEMFIELD_USE_OPENCL
        fFastMultipoleFieldSolverOpenCL =
            new KFMElectrostaticFastMultipoleFieldSolver_OpenCL(fIntegratorPolicy.CreateOpenCLConfig(),
                                                                *oclContainer,
                                                                *fTree);
        return;
#endif
    }

    fFastMultipoleFieldSolver = new KFMElectrostaticFastMultipoleFieldSolver(container, *fTree);
    return;
}

double KElectricFastMultipoleFieldSolver::PotentialCore(const KPosition& P) const
{
    if (fUseOpenCL) {
#ifdef KEMFIELD_USE_OPENCL
        return fFastMultipoleFieldSolverOpenCL->Potential(P);
#endif
    }
    return fFastMultipoleFieldSolver->Potential(P);
}

KThreeVector KElectricFastMultipoleFieldSolver::ElectricFieldCore(const KPosition& P) const
{
    if (fUseOpenCL) {
#ifdef KEMFIELD_USE_OPENCL
        return fFastMultipoleFieldSolverOpenCL->ElectricField(P);
#endif
    }
    return fFastMultipoleFieldSolver->ElectricField(P);
}


void KElectricFastMultipoleFieldSolver::UseOpenCL(bool choice)
{
    if (choice == true) {
#ifdef KEMFIELD_USE_OPENCL
        fUseOpenCL = choice;
        return;
#endif
        KEMField::cout
            << "cannot use opencl in fast multipole without kemfield being built with opencl, using defaults."
            << KEMField::endl;
    }
    fUseOpenCL = false;
    return;
}


} /* namespace KEMField */
