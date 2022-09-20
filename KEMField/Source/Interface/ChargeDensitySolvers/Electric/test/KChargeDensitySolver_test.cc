/*
 * KChargeDensitySolver_test.cc
 *
 *  Created on: 25 Jun 2015
 *      Author: wolfgang
 */

#include "KChargeDensitySolver.hh"
#include "KGaussianEliminationChargeDensitySolver.hh"

using namespace KGeoBag;
using namespace KEMField;

int main(int /*argc*/, char** /*args*/)
{
    std::shared_ptr<KChargeDensitySolver> ptr(nullptr);
    {
        auto* raw = new KGaussianEliminationChargeDensitySolver;
        std::shared_ptr<KChargeDensitySolver> ptr1(raw);
        ptr = ptr1;
    }
    auto smartContainer = std::make_shared<KSurfaceContainer>();
    ptr->Initialize(*smartContainer);
    return 0;
}
