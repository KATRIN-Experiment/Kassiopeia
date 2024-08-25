#include "KSTimed.h"

namespace Kassiopeia
{

void KSTimed::StartTiming() {
    fTimeStart = std::chrono::steady_clock::now();
    fProcessingDuration = -1;
}

void KSTimed::EndTiming() {
    std::chrono::steady_clock::time_point tTimeEnd = std::chrono::steady_clock::now();
    fProcessingDuration = std::chrono::duration_cast<std::chrono::duration<double>>(tTimeEnd - fTimeStart).count();
}

}
