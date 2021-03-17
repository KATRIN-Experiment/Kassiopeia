#include "KFMElectrostaticNodeWorkScoreCalculator.hh"

#include <cmath>

namespace KEMField
{

KFMElectrostaticNodeWorkScoreCalculator::KFMElectrostaticNodeWorkScoreCalculator()
{
    fNTerms = 0;
    fDivisions = 0;
    fZeroMaskSize = 0;

    fNMultipoleNodes = 0;
    fNonLeafMultipoleNodes = 0;
    fNPrimaryNodes = 0;
    fNSources = 0;
    fNCollocationPoints = 0;
    fNSparseMatrixElements = 0;
    fNodeScore = 0;

    //weights for scoring
    //TODO tune these weights
    fAlpha = 0.0;
    fBeta = 0.0;
    fGamma = 1.0;
    fDelta = 0.0;
    fEpsilon = 0.0;
    fTheta = 1.0;

    fRecursiveActor.SetOperationalActor(&fSingleNodeActor);
};

KFMElectrostaticNodeWorkScoreCalculator::~KFMElectrostaticNodeWorkScoreCalculator() = default;
;

void KFMElectrostaticNodeWorkScoreCalculator::ApplyAction(KFMNode<KFMElectrostaticNodeObjects>* node)
{
    fNMultipoleNodes = 0;
    fNonLeafMultipoleNodes = 0;
    fNPrimaryNodes = 0;
    fNSources = 0;
    fNCollocationPoints = 0;
    fNSparseMatrixElements = 0;
    fNodeScore = 0;

    fSingleNodeActor.Reset();

    if (node != nullptr) {
        //we need to recursively vist this node and its children
        //to calculate the number of operations required to evaluate
        //the field at all of the collocation points contained in the
        //relevant region
        fRecursiveActor.ApplyAction(node);

        fNMultipoleNodes = fSingleNodeActor.GetNMultipoleNodes();
        fNonLeafMultipoleNodes = fSingleNodeActor.GetNonLeafMultipoleNodes();
        fNPrimaryNodes = fSingleNodeActor.GetNPrimaryNodes();
        fNSources = fSingleNodeActor.GetNSources();
        fNCollocationPoints = fSingleNodeActor.GetNCollocationPoints();
        fNSparseMatrixElements = fSingleNodeActor.GetNSparseMatrixElements();

        CalculateFinalScore();
    }
}

void KFMElectrostaticNodeWorkScoreCalculator::CalculateFinalScore()
{
    fNodeScore = 0.0;

    //node score is approximated by the cost to evaluate the sparse matrix multiplication
    //plust the cose to perform the M2L transformations
    //(we ignore M2M and L2L and L2P for now as they are not dominant)

    //add cost due to the multipole calculation
    fNodeScore += fAlpha * fNTerms * fNSources;
    //add cost due to M2M translation
    fNodeScore += fBeta * fNTerms * fNTerms * fNMultipoleNodes;
    //add cost due to M2L transform
    fNodeScore += fGamma * fNTerms * fNTerms * fNonLeafMultipoleNodes;
    //add cost due to L2L transform
    fNodeScore += fDelta * fNTerms * fNTerms * fNPrimaryNodes;
    //add cost due to local coeff field evaluation
    fNodeScore += fEpsilon * fNTerms * fNCollocationPoints;
    //add cost due to sparse matrix evaluation
    fNodeScore += fTheta * fNSparseMatrixElements;
}


}  // namespace KEMField
