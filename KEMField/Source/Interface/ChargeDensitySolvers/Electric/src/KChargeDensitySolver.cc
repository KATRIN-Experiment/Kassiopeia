/*
 * KChargeDensitySolver.cc
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 */

#include "KChargeDensitySolver.hh"

#include "KEMFileInterface.hh"
#include "KMD5HashGenerator.hh"
#include "KProjectionSolver.hh"
#include "KSVDSolver.hh"
#include "KSuperpositionSolver.hh"
#include "KTypeManipulation.hh"

#ifdef KEMFIELD_USE_ROOT
#include "KEMRootSVDSolver.hh"
#endif

#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KBoundaryIntegralVector.hh"
#include "KEMCoreMessage.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KIterativeStateWriter.hh"
#include "KSquareMatrix.hh"

#ifdef KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
#ifndef MPI_SINGLE_PROCESS
#define MPI_SINGLE_PROCESS if (KEMField::KMPIInterface::GetInstance()->GetProcess() == 0)
#endif
#else
#ifndef MPI_SINGLE_PROCESS
#define MPI_SINGLE_PROCESS if (true)
#endif
#endif

using namespace std;

namespace KEMField
{

void KChargeDensitySolver::SetHashProperties(unsigned int maskedBits, double hashThreshold)
{
    fHashMaskedBits = maskedBits;
    fHashThreshold = hashThreshold;
}

bool KChargeDensitySolver::FindSolution(double aThreshold, KSurfaceContainer& aContainer)
{
    // compute shape hash
    KMD5HashGenerator tShapeHashGenerator;
    tShapeHashGenerator.MaskedBits(fHashMaskedBits);
    tShapeHashGenerator.Threshold(fHashThreshold);
    tShapeHashGenerator.Omit(KEMField::Type2Type<KElectrostaticBasis>());
    tShapeHashGenerator.Omit(Type2Type<KBoundaryType<KElectrostaticBasis, KDirichletBoundary>>());
    string tShapeHash = tShapeHashGenerator.GenerateHash(aContainer);

    kem_cout_debug("<shape> hash is <" << tShapeHash << ">" << eom);

    // compute shape+boundary hash
    KMD5HashGenerator tShapeBoundaryHashGenerator;
    tShapeBoundaryHashGenerator.MaskedBits(fHashMaskedBits);
    tShapeBoundaryHashGenerator.Threshold(fHashThreshold);
    tShapeBoundaryHashGenerator.Omit(Type2Type<KElectrostaticBasis>());
    string tShapeBoundaryHash = tShapeBoundaryHashGenerator.GenerateHash(aContainer);

    kem_cout_debug("<shape+boundary> hash is <" << tShapeBoundaryHash << ">" << eom);

    vector<string> tLabels;
    unsigned int tCount;
    bool tSolution;

    // compose residual threshold labels for shape and shape+boundary
    tLabels.clear();
    tLabels.push_back(KResidualThreshold::Name());
    tLabels.push_back(tShapeHash);
    tLabels.push_back(tShapeBoundaryHash);

    kem_cout_debug("charge density solver looking for existing solution ..." << eom);

    // find matching residual thresholds

    tCount = KEMFileInterface::GetInstance()->NumberWithLabels(tLabels);
    kem_cout_debug("found <" << tCount << "> solutions that match the <shape> and <shape+boundary> hashes" << eom);

    if (tCount > 0) {
        KResidualThreshold tResidualThreshold;
        KResidualThreshold tMinResidualThreshold;

        for (unsigned int i = 0; i < tCount; i++) {
            KEMFileInterface::GetInstance()->FindByLabels(tResidualThreshold, tLabels, i);
            kem_cout_debug("found threshold <" << tResidualThreshold.fResidualThreshold << ">" << eom);

            if (tResidualThreshold < tMinResidualThreshold) {
                kem_cout_debug("found minimum solution <" << tResidualThreshold.fGeometryHash << "> with threshold <"
                                                          << tResidualThreshold.fResidualThreshold << ">" << eom);

                tMinResidualThreshold = tResidualThreshold;
            }
        }

        kem_cout_debug("global minimum solution <" << tMinResidualThreshold.fGeometryHash << "> with threshold <"
                                                   << tResidualThreshold.fResidualThreshold << ">" << eom);

        tSolution = false;
        string tSolutionFilename;
        auto* tNewContainer = new KSurfaceContainer();
        KEMFileInterface::GetInstance()->FindByHash(*tNewContainer,
                                                    tMinResidualThreshold.fGeometryHash,
                                                    tSolution,
                                                    tSolutionFilename);

        // restore names/tags that are not stored with the container
        for (size_t i = 0; i < aContainer.size(); ++i) {
            if (tNewContainer->at(i)->GetShape()->Centroid() == aContainer[i]->GetShape()->Centroid()) {
                tNewContainer->at(i)->GetShape()->SetName(aContainer[i]->GetShape()->GetName());
                tNewContainer->at(i)->GetShape()->SetTagsFrom(aContainer[i]->GetShape());
            }
        }

        if (tMinResidualThreshold.fResidualThreshold <= aThreshold) {
            MPI_SINGLE_PROCESS
            kem_cout() << "previously computed charge density solution found in file <" << tSolutionFilename << ">"
                       << eom;
            tSolution = true;
        }

        if (tSolution == true) {
            aContainer = std::move(*tNewContainer);
            return true;
        }
    }

    // compose residual threshold labels for shape
    tLabels.clear();
    tLabels.push_back(KResidualThreshold::Name());
    tLabels.push_back(tShapeHash);

    // find residual thresholds for geometry
    tCount = KEMFileInterface::GetInstance()->NumberWithLabels(tLabels);

    kem_cout_debug("found <" << tCount << "> that match <shape> hash" << eom);

    if (tCount > 0) {
#ifdef KEMFIELD_USE_ROOT
        KSuperpositionSolver<double, KEMRootSVDSolver> tSuperpositionSolver;
#else
        KSuperpositionSolver<double, KSVDSolver>
            tSuperpositionSolver;  // this doesn't seem to work even if gsl is enabled
#endif

        // The integrator type (numeric, analytic) shouldn't matter here.
        // We just need the access to the boundary elements
        //potential/permittivity change and  charge density values.
        KElectrostaticBoundaryIntegrator tIntegrator{KEBIFactory::MakeDefault()};
        KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> tVector(aContainer, tIntegrator);
        KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> tSolutionVector(aContainer, tIntegrator);

        KResidualThreshold tResidualThreshold;
        vector<KSurfaceContainer*> tContainers;
        vector<KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator>*> tSolutionVectors;
        vector<KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator>*> tVectors;

        vector<string> tSolutionFilenames = {""};
        for (unsigned int tIndex = 0; tIndex < tCount; tIndex++) {
            KEMFileInterface::GetInstance()->FindByLabels(tResidualThreshold, tLabels, tIndex);

            kem_cout_debug("found threshold <" << tResidualThreshold.fResidualThreshold << ">" << eom);

            if (tResidualThreshold.fResidualThreshold <= aThreshold) {
                kem_cout_debug("adding solution <" << tResidualThreshold.fGeometryHash << "> with threshold <"
                                                   << tResidualThreshold.fResidualThreshold << ">" << eom);

                auto* tNewContainer = new KSurfaceContainer();
                KEMFileInterface::GetInstance()->FindByHash(*tNewContainer,
                                                            tResidualThreshold.fGeometryHash,
                                                            tSolution,
                                                            tSolutionFilenames.back());

                auto* tNewVector =
                    new KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator>(*tNewContainer, tIntegrator);
                auto* tNewSolutionVector =
                    new KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator>(*tNewContainer, tIntegrator);

                tContainers.push_back(tNewContainer);
                tVectors.push_back(tNewVector);
                tSolutionVectors.push_back(tNewSolutionVector);
                tSuperpositionSolver.AddSolvedSystem(*tNewSolutionVector, *tNewVector);
            }

            tSolutionFilenames.push_back("");
        }

        tSolution = false;
        if (tSuperpositionSolver.SolutionSpaceIsSpanned(tVector)) {
            tSuperpositionSolver.ComposeSolution(tSolutionVector);
            kem_cout() << "superposition of previously computed charge density solutions found" << eom;
            tSolution = true;
        }

        for (auto& tContainer : tContainers) {
            delete tContainer;
        }
        for (auto& tSolutionVector : tSolutionVectors) {
            delete tSolutionVector;
        }
        for (auto& tVector : tVectors) {
            delete tVector;
        }

        if (tSolution == true) {
            return true;
        }
    }

    kem_cout(eInfo) << "no previously computed charge density solution found." << eom;
    return false;
}
void KChargeDensitySolver::SaveSolution(double aThreshold, KSurfaceContainer& aContainer) const
{
    // compute hash of the bare geometry
    KMD5HashGenerator tShapeHashGenerator;
    tShapeHashGenerator.MaskedBits(fHashMaskedBits);
    tShapeHashGenerator.Threshold(fHashThreshold);
    tShapeHashGenerator.Omit(Type2Type<KElectrostaticBasis>());
    tShapeHashGenerator.Omit(Type2Type<KBoundaryType<KElectrostaticBasis, KDirichletBoundary>>());
    string tShapeHash = tShapeHashGenerator.GenerateHash(aContainer);

    kem_cout_debug("<shape> hash is <" << tShapeHash << ">" << eom);

    // compute hash of the boundary values on the bare geometry
    KMD5HashGenerator tShapeBoundaryHashGenerator;
    tShapeBoundaryHashGenerator.MaskedBits(fHashMaskedBits);
    tShapeBoundaryHashGenerator.Threshold(fHashThreshold);
    tShapeBoundaryHashGenerator.Omit(Type2Type<KElectrostaticBasis>());
    string tShapeBoundaryHash = tShapeBoundaryHashGenerator.GenerateHash(aContainer);

    kem_cout_debug("<shape+boundary> hash is <" << tShapeBoundaryHash << ">" << eom);

    // compute hash of solution with boundary values on the bare geometry
    KMD5HashGenerator tShapeBoundarySolutionHashGenerator;
    string tShapeBoundarySolutionHash = tShapeBoundarySolutionHashGenerator.GenerateHash(aContainer);

    kem_cout_debug("<shape+boundary+solution> hash is <" << tShapeBoundarySolutionHash << ">" << eom);

    // create label set for summary object
    string tThresholdBase(KResidualThreshold::Name());
    string tThresholdName = tThresholdBase + string("_") + tShapeBoundarySolutionHash;
    vector<string> tThresholdLabels;
    tThresholdLabels.push_back(tThresholdBase);
    tThresholdLabels.push_back(tShapeHash);
    tThresholdLabels.push_back(tShapeBoundaryHash);
    tThresholdLabels.push_back(tShapeBoundarySolutionHash);

    // write summary object;
    KResidualThreshold tResidualThreshold;
    tResidualThreshold.fResidualThreshold = aThreshold;
    tResidualThreshold.fGeometryHash = tShapeBoundarySolutionHash;
    MPI_SINGLE_PROCESS
    {
        KEMFileInterface::GetInstance()->Write(tResidualThreshold, tThresholdName, tThresholdLabels);
    }
    // create label set for container object
    string tContainerBase(KSurfaceContainer::Name());
    string tContainerName = tContainerBase + string("_") + tShapeBoundarySolutionHash;
    vector<string> tContainerLabels;
    tContainerLabels.push_back(tContainerBase);
    tContainerLabels.push_back(tShapeBoundarySolutionHash);

    // write container object
    MPI_SINGLE_PROCESS
    {
        KEMFileInterface::GetInstance()->Write(aContainer, tContainerName, tContainerLabels);
    }

    return;
}

}  // namespace KEMField
