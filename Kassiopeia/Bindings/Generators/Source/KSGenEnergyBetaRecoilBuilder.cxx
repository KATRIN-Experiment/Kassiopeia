//
// Created by wdconinc on 13.02.20.
//

#include "KSGenEnergyBetaRecoilBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenEnergyBetaRecoilBuilder::~KComplexElement() = default;

STATICINT sKSGenEnergyBetaRecoilStructure = KSGenEnergyBetaRecoilBuilder::Attribute<std::string>("name") +
                                            KSGenEnergyBetaRecoilBuilder::Attribute<double>("min_energy") +
                                            KSGenEnergyBetaRecoilBuilder::Attribute<double>("max_energy");

STATICINT sKSGenEnergyBetaRecoil = KSRootBuilder::ComplexElement<KSGenEnergyBetaRecoil>("ksgen_energy_beta_recoil");

}  // namespace katrin
