//
// Created by Nikolaus Trost on 07.05.15.
//
#include "KSGenLUniformMaxNBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenLUniformMaxNBuilder::~KComplexElement() = default;

STATICINT sKSGenLUniformMaxNStructure = KSGenLUniformMaxNBuilder::Attribute<std::string>("name");

STATICINT sKSGenLUniformMaxN = KSRootBuilder::ComplexElement<KSGenLUniformMaxN>("ksgen_l_uniform_max_n");
}  // namespace katrin
