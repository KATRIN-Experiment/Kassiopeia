#include "KApplicationRunner.h"

#include "KInitializationMessage.hh"

#include <exception>

namespace katrin
{

KApplicationRunner::KApplicationRunner() = default;

KApplicationRunner::KApplicationRunner(const KApplicationRunner&) = default;

KApplicationRunner::~KApplicationRunner()
{
    for (auto& app : fApplications)
        delete app;
}

bool KApplicationRunner::Execute()
{
    bool success = true;

    for (auto& app : fApplications) {
        try {
            bool tSuccess = app->Execute();
            success = (success && tSuccess);
            if (!tSuccess)
                initmsg(eWarning) << "Application " << app->GetName() << " failed " << eom;
        }
        catch (std::exception& e) {
            initmsg(eWarning) << "Application " << app->GetName() << " failed with exception: " << ret;
            initmsg(eWarning) << e.what() << eom;
            continue;
        }
    }

    return success;
}

void KApplicationRunner::AddApplication(KApplication* tApplication)
{
    fApplications.push_back(tApplication);
}

}  // namespace katrin
