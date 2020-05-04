#ifndef KFMElectrostaticParametersConfiguration_HH__
#define KFMElectrostaticParametersConfiguration_HH__

#include "KFMElectrostaticParameters.hh"
#include "KSAStructuredASCIIHeaders.hh"

#include <string>

namespace KEMField
{

/*
*
*@file KFMElectrostaticParametersConfiguration.hh
*@class KFMElectrostaticParametersConfiguration
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Feb 8 14:16:47 EST 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticParametersConfiguration : public KSAInputOutputObject
{
  public:
    KFMElectrostaticParametersConfiguration()
    {
        fVerbosity = 3;
        fStrategy = 0;
        fTopLevelDivisions = 3;
        fDivisions = 3;
        fDegree = 0;
        fZeroMaskSize = 1;
        fMaxTreeLevel = 3;
        fInsertionRatio = 4.0 / 3.0;
        fUseRegionEstimation = 1;
        fUseCaching = 0;
        fRegionExpansionFactor = 1.1;
        fWorldCenterX = 0.0;
        fWorldCenterY = 0.0;
        fWorldCenterZ = 0.0;
        fWorldLength = 0.0;
        fAllowedNumber = 1;
        fAllowedFraction = 1;
        fBiasDegree = 1;
    }

    ~KFMElectrostaticParametersConfiguration() override
    {
        ;
    };

    int GetVerbosity() const
    {
        return fVerbosity;
    };
    void SetVerbosity(const int& n)
    {
        fVerbosity = n;
    };

    int GetStrategy() const
    {
        return fStrategy;
    };
    void SetStrategy(const int& n)
    {
        fStrategy = n;
    };

    int GetDivisions() const
    {
        return fDivisions;
    };
    void SetDivisions(const int& d)
    {
        fDivisions = d;
    };

    int GetTopLevelDivisions() const
    {
        return fTopLevelDivisions;
    };
    void SetTopLevelDivisions(const int& d)
    {
        fTopLevelDivisions = d;
    };

    int GetDegree() const
    {
        return fDegree;
    };
    void SetDegree(const int& deg)
    {
        fDegree = deg;
    };

    int GetZeroMaskSize() const
    {
        return fZeroMaskSize;
    };
    void SetZeroMaskSize(const int& z)
    {
        fZeroMaskSize = z;
    };

    int GetMaxTreeLevel() const
    {
        return fMaxTreeLevel;
    };
    void SetMaxTreeLevel(const int& t)
    {
        fMaxTreeLevel = t;
    };

    int GetUseRegionEstimation() const
    {
        return fUseRegionEstimation;
    };
    void SetUseRegionEstimation(const int& r)
    {
        fUseRegionEstimation = r;
    };

    double GetRegionExpansionFactor() const
    {
        return fRegionExpansionFactor;
    };
    void SetRegionExpansionFactor(const double& d)
    {
        fRegionExpansionFactor = d;
    };

    double GetInsertionRatio() const
    {
        return fInsertionRatio;
    };
    void SetInsertionRatio(const double& d)
    {
        fInsertionRatio = d;
    };

    int GetUseCaching() const
    {
        return fUseCaching;
    };
    void SetUseCaching(const int& r)
    {
        fUseCaching = r;
    };

    double GetWorldCenterX() const
    {
        return fWorldCenterX;
    };
    void SetWorldCenterX(const double& d)
    {
        fWorldCenterX = d;
    };

    double GetWorldCenterY() const
    {
        return fWorldCenterY;
    };
    void SetWorldCenterY(const double& d)
    {
        fWorldCenterY = d;
    };

    double GetWorldCenterZ() const
    {
        return fWorldCenterZ;
    };
    void SetWorldCenterZ(const double& d)
    {
        fWorldCenterZ = d;
    };

    double GetWorldLength() const
    {
        return fWorldLength;
    };
    void SetWorldLength(const double& d)
    {
        fWorldLength = d;
    };

    int GetAllowedNumber() const
    {
        return fAllowedNumber;
    };
    void SetAllowedNumber(const int& r)
    {
        fAllowedNumber = r;
    };

    double GetAllowedFraction() const
    {
        return fAllowedFraction;
    };
    void SetAllowedFraction(const double& r)
    {
        fAllowedFraction = r;
    };

    double GetBiasDegree() const
    {
        return fBiasDegree;
    };
    void SetBiasDegree(const double& r)
    {
        fBiasDegree = r;
    };

    KFMElectrostaticParameters GetParameters() const
    {
        KFMElectrostaticParameters params;

        params.strategy = fStrategy;
        params.top_level_divisions = fTopLevelDivisions;
        params.divisions = fDivisions;
        params.degree = fDegree;
        params.zeromask = fZeroMaskSize;
        params.maximum_tree_depth = fMaxTreeLevel;
        params.insertion_ratio = fInsertionRatio;
        params.verbosity = fVerbosity;

        params.use_region_estimation = false;
        if (fUseRegionEstimation != 0) {
            params.use_region_estimation = true;
        }

        params.use_caching = false;
        if (fUseCaching != 0) {
            params.use_caching = true;
        }

        params.region_expansion_factor = fRegionExpansionFactor;

        if (params.use_region_estimation) {
            params.world_center_x = 0.0;
            params.world_center_y = 0.0;
            params.world_center_z = 0.0;
            params.world_length = 0.0;
        }
        else {
            params.world_center_x = fWorldCenterX;
            params.world_center_y = fWorldCenterY;
            params.world_center_z = fWorldCenterZ;
            params.world_length = fWorldLength;
        }


        if (params.strategy == KFMSubdivisionStrategy::Guided) {
            params.allowed_number = fAllowedNumber;
            params.allowed_fraction = fAllowedFraction;
        }
        else {
            params.allowed_number = 1;
            params.allowed_fraction = 1.0;
        }

        if (params.strategy == KFMSubdivisionStrategy::Balanced) {
            params.bias_degree = fBiasDegree;
        }
        else {
            params.bias_degree = 1;
        }

        return params;
    }

    void DefineOutputNode(KSAOutputNode* node) const override
    {
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, Verbosity, int);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, Strategy, int);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, TopLevelDivisions, int);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, Divisions, int);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, Degree, int);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, ZeroMaskSize, int);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, MaxTreeLevel, int);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, UseRegionEstimation, int);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, UseCaching, int);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, RegionExpansionFactor, double);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, InsertionRatio, double);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, WorldCenterX, double);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, WorldCenterY, double);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, WorldCenterZ, double);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, WorldLength, double);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, AllowedNumber, int);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, AllowedFraction, double);
        AddKSAOutputFor(KFMElectrostaticParametersConfiguration, BiasDegree, double);
    }

    void DefineInputNode(KSAInputNode* node) override
    {
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, Strategy, int);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, Verbosity, int);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, TopLevelDivisions, int);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, Divisions, int);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, Degree, int);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, ZeroMaskSize, int);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, MaxTreeLevel, int);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, UseRegionEstimation, int);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, UseCaching, int);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, RegionExpansionFactor, double);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, InsertionRatio, double);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, WorldCenterX, double);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, WorldCenterY, double);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, WorldCenterZ, double);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, WorldLength, double);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, AllowedNumber, int);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, AllowedFraction, double);
        AddKSAInputFor(KFMElectrostaticParametersConfiguration, BiasDegree, double);
    }

    virtual std::string ClassName() const
    {
        return std::string("KFMElectrostaticParametersConfiguration");
    };

  protected:
    int fStrategy;
    int fVerbosity;
    int fTopLevelDivisions;
    int fDivisions;
    int fDegree;
    int fZeroMaskSize;
    int fMaxTreeLevel;
    int fUseRegionEstimation;
    int fUseCaching;
    double fRegionExpansionFactor;
    double fInsertionRatio;
    double fWorldCenterX;
    double fWorldCenterY;
    double fWorldCenterZ;
    double fWorldLength;
    int fAllowedNumber;
    double fAllowedFraction;
    double fBiasDegree;
};

DefineKSAClassName(KFMElectrostaticParametersConfiguration);

}  // namespace KEMField

#endif /* KFMElectrostaticParametersConfiguration_HH__ */
