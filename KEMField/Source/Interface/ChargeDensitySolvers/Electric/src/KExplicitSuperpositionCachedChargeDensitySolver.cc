/*
 * ExplicitSuperpositionChargeDensitySolver.cc
 *
 *  Created on: 27 Jun 2016
 *      Author: wolfgang
 */

#include "KExplicitSuperpositionCachedChargeDensitySolver.hh"

#include "KBoundaryIntegralSolutionVector.hh"
#include "KEMFileInterface.hh"
#include "KEMSimpleException.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KIterativeStateWriter.hh"
#include "KStringUtils.h"

#include "KEMCoreMessage.hh"
#include "KStringUtils.h"

namespace KEMField
{

KExplicitSuperpositionCachedChargeDensitySolver::KExplicitSuperpositionCachedChargeDensitySolver()
{
    fName = "";
    fNames.clear();
    fScaleFactors.clear();
    fHashLabels.clear();
}

KExplicitSuperpositionCachedChargeDensitySolver::~KExplicitSuperpositionCachedChargeDensitySolver() = default;

void KExplicitSuperpositionCachedChargeDensitySolver::InitializeCore(KSurfaceContainer& container)
{
    bool tSolution = false;
    bool tCompleteSolution = false;

    if ((fScaleFactors.empty()) && (fHashLabels.empty())) {
        throw KEMSimpleException(
            "must provide a set of scale factors and hash labels for explicit cached bem solution");
    }

    //create a solution vector

    // The integrator type (numeric, analytic) shouldn't matter here.
    // We just need the access to the boundary elements
    //potential/permittivity change and  charge density values.
    KElectrostaticBoundaryIntegrator integrator{KEBIFactory::MakeDefault()};
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(container, integrator);

    //zero out the solution vector
    unsigned int geometry_dim = x.Dimension();
    for (unsigned int i = 0; i < x.Dimension(); i++) {
        x[i] = 0;
    }

    kem_cout_debug("explicit superposition charge density solver <" << fName << "> looking for "
                   << geometry_dim << "-dimensional solution ..." << eom);

    std::vector<std::string> tSolutionFilenames = {""};
    unsigned int found_count = 0;
    KResidualThreshold tMaxResidualThreshold;
    tMaxResidualThreshold.fResidualThreshold = 0;
    for (unsigned int n = 0; n < fScaleFactors.size(); n++) {
        tSolution = false;

        auto* tempContainer = new KSurfaceContainer();
        KEMFileInterface::GetInstance()->FindByHash(*tempContainer,
                                                    fHashLabels[n],
                                                    tSolution,
                                                    tSolutionFilenames.back());

        KResidualThreshold tResidualThreshold;
        std::vector<std::string> labels;
        labels.push_back(fHashLabels[n]);
        KEMFileInterface::GetInstance()->FindByLabels(tResidualThreshold, labels);

        if (tMaxResidualThreshold.fResidualThreshold < tResidualThreshold.fResidualThreshold) {
            tMaxResidualThreshold.fResidualThreshold = tResidualThreshold.fResidualThreshold;
        }

        if (tSolution) {
            KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> temp_x(*tempContainer, integrator);
            //We are using the geometry dimension to serve as a very crude hash mechanism
            //since, unfortunately KEMField's MD5 hash mechanism is machine dependent and cannot match geometries
            //that were constructed on different machines. This is not a perfect check, since the user could
            //do something that would make the geometries have the same size but not
            //match geometrically (i.e coordinate transformation).
            //However, this check is better than nothing, and should catch obvious errors.
            if (geometry_dim == temp_x.Dimension()) {
                kem_cout(eDebug) << "adding scaled solution with factor <" << fScaleFactors[n]
                                 << "> for geometry with name <" << fNames[n] << ">" << eom;
                //sum in the scaled contribution from this solution
                double scale = fScaleFactors[n];
                for (unsigned int i = 0; i < x.Dimension(); i++) {
                    x[i] += scale * temp_x[i];
                }
                found_count++;
            }
            else {
                throw KEMSimpleException("problem dimension does not match that of geometry <" + fNames[n] +
                                         "> with hash label <" + fHashLabels[n] + ">.");
            }
            delete tempContainer;
        }
        else {
            throw KEMSimpleException("no matching solution with hash <" + fHashLabels[n] + "> for <" + fNames[n] +
                                     ">.");
        }

        tSolutionFilenames.push_back("");
    }

    if (found_count == fScaleFactors.size()) {
        tCompleteSolution = true;
        kem_cout() << "superposition of previously computed charge density solutions found in " << tSolutionFilenames.size() << " files" << eom;
        SaveSolution(tMaxResidualThreshold.fResidualThreshold, container);
    }

    if (tCompleteSolution == false) {
        throw KEMSimpleException("could not find needed set of cached bem solutions in directory <" +
                                 KEMFileInterface::GetInstance()->ActiveDirectory() + ">");
    }
}


} /* namespace KEMField */
