/*
 * KFastMultipoleMatrixGenerator.hh
 *
 *  Created on: 17 Aug 2015
 *      Author: wolfgang
 */

#ifndef KFASTMULTIPOLEMATRIXGENERATOR_HH_
#define KFASTMULTIPOLEMATRIXGENERATOR_HH_

#include "KBoundaryMatrixGenerator.hh"
#include "KFMElectrostaticTypes.hh"

namespace KEMField
{

class KFastMultipoleMatrixGenerator : public KBoundaryMatrixGenerator<KFMElectrostaticTypes::ValueType>
{
  public:
    typedef KFMElectrostaticTypes::ValueType ValueType;

    KFastMultipoleMatrixGenerator();
    ~KFastMultipoleMatrixGenerator() override;

    KSmartPointer<KSquareMatrix<ValueType>> Build(const KSurfaceContainer& container) const override;

    void SetDirectIntegrator(const KElectrostaticBoundaryIntegrator& integrator)
    {
        fDirectIntegrator = integrator;
    }

    void SetStrategy(int strategy)
    {
        fParameters.strategy = strategy;
    }

    void SetDegree(unsigned int degree)
    {
        fParameters.degree = degree;
    }

    void SetDivisions(unsigned int divisions)
    {
        fParameters.divisions = divisions;
    }

    void SetInsertionRatio(double insertionRatio)
    {
        fParameters.insertion_ratio = insertionRatio;
    }

    void SetMaximumTreeDepth(unsigned int maximumTreeDepth)
    {
        fParameters.maximum_tree_depth = maximumTreeDepth;
    }

    void SetRegionExpansionFactor(double regionExpansionFactor)
    {
        fParameters.region_expansion_factor = regionExpansionFactor;
    }

    void SetTopLevelDivisions(unsigned int topLevelDivisions)
    {
        fParameters.top_level_divisions = topLevelDivisions;
    }

    void SetUseCaching(bool useCaching)
    {
        fParameters.use_caching = useCaching;
    }

    void SetUseRegionEstimation(bool useRegionEstimation)
    {
        fParameters.use_region_estimation = useRegionEstimation;
    }

    void SetVerbosity(unsigned int verbosity)
    {
        fParameters.verbosity = verbosity;
    }

    void SetWorldCenter(const KPosition& world_center)
    {
        fParameters.world_center_x = world_center.X();
        fParameters.world_center_y = world_center.Y();
        fParameters.world_center_z = world_center.Z();
    }

    void SetWorldLength(double worldLength)
    {
        fParameters.world_length = worldLength;
    }

    void SetZeromask(unsigned int zeromask)
    {
        fParameters.zeromask = zeromask;
    }

    void SetBiasDegree(double bias_degree)
    {
        fParameters.bias_degree = bias_degree;
    }

    void SetAllowedFraction(double allowed_fraction)
    {
        fParameters.allowed_fraction = allowed_fraction;
    }

    void SetAllowedNumber(double allowed_number)
    {
        fParameters.allowed_number = allowed_number;
    }


  private:
    KSmartPointer<KFMElectrostaticTypes::FastMultipoleMatrix>
    CreateMatrix(const KSurfaceContainer& surfaceContainer,
                 KSmartPointer<KFMElectrostaticTypes::FastMultipoleEBI>) const;

    KFMElectrostaticParameters fParameters;
    KElectrostaticBoundaryIntegrator fDirectIntegrator;
};

} /* namespace KEMField */

#endif /* KFASTMULTIPOLEMATRIXGENERATOR_HH_ */
