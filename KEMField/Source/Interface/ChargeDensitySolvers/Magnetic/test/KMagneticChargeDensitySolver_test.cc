/*
 * KMagneticChargeDensitySolver_test.cc
 *
 *  Created on: 7 Jun 2025
 *      Author: pslocum
 */

#include "KMagneticChargeDensitySolver.hh"
#include "KBiotSavartChargeDensitySolver.hh"

using namespace KGeoBag;
using namespace KEMField;

int main(int /*argc*/, char** /*args*/)
{
    std::shared_ptr<KMagneticChargeDensitySolver> ptr(nullptr);
    {
        auto* raw = new KBiotSavartChargeDensitySolver;
        std::shared_ptr<KMagneticChargeDensitySolver> ptr1(raw);
        ptr = ptr1;
    }
    auto smartContainer = std::make_shared<KSurfaceContainer>();
    ptr->Initialize(*smartContainer);
    return 0;
}
