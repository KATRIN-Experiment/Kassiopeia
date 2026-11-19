#include "KSTrajControlEnergyBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajControlEnergyBuilder::~KComplexElement() = default;

STATICINT sKSTrajControlEnergyStructure =
    //! the name of this control.
    KSTrajControlEnergyBuilder::Attribute<std::string>("name") +
    //! the energy conservation violation threshold below which larger temporal steps are suggested.
    KSTrajControlEnergyBuilder::Attribute<double>("lower_limit") +
    //! the maximum tolerable energy conservation violation per step.
    KSTrajControlEnergyBuilder::Attribute<double>("upper_limit") +
    KSTrajControlEnergyBuilder::Attribute<double>("min_length") +
    KSTrajControlEnergyBuilder::Attribute<double>("max_length") +
    KSTrajControlEnergyBuilder::Attribute<double>("initial_step") +
    KSTrajControlEnergyBuilder::Attribute<double>("adjustment") +
    KSTrajControlEnergyBuilder::Attribute<double>("adjustment_up") +
    KSTrajControlEnergyBuilder::Attribute<double>("adjustment_down") +
    KSTrajControlEnergyBuilder::Attribute<double>("step_rescale");

STATICINT sToolboxKSTrajControlEnergy = KSRootBuilder::ComplexElement<KSTrajControlEnergy>("kstraj_control_energy");

}  // namespace katrin
