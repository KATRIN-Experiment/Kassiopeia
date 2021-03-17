#include "KEMTicker.hh"

#include "KEMCoreMessage.hh"

#include <cmath>
#include <string>

namespace KEMField
{
void KTicker::StartTicker(double goal)
{
    fGoal = goal;
    fCounter = 0;
    if (fabs(goal) < 1.e-10)
        fNoGoal = true;

    if (fNoGoal)
        kem_cout(eNormal) << "[ ]" << reom;
    else
        kem_cout(eNormal) << "[0%]" << reom;
}

void KTicker::Tick(double progress) const
{
    std::string tickMark;

    if (fNoGoal) {
        if (fCounter % 4 == 0)
            tickMark = "[|]";
        else if (fCounter % 4 == 1)
            tickMark = "[/]";
        else if (fCounter % 4 == 2)
            tickMark = "[-]";
        else
            tickMark = "[\\]";
        fCounter++;
    }
    else {
        auto counter = (unsigned int) std::fabs(progress / fGoal * 100.);
        if (counter != fCounter) {
            fCounter = counter;
            std::stringstream s;
            s << "[";
            if (counter < 10)
                s << "  ";
            else if (counter < 100)
                s << " ";
            s << counter << "%]  ";
            tickMark = s.str();
        }
    }

    kem_cout(eNormal) << tickMark << reom;
}

void KTicker::EndTicker() const
{
    kem_cout(eNormal) << "[100%]" << eom;
}
}  // namespace KEMField
