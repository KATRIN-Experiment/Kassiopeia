#include "KEMTicker.hh"

#include <cmath>

namespace KEMField
{
  void KTicker::StartTicker(double goal)
  {
    fGoal = goal;
    fCounter = 0;
    if (fabs(goal)<1.e-10)
      fNoGoal = true;
  }

  void KTicker::Tick(double progress) const
  {
    std::string tickMark;

    if (fNoGoal)
    {
      if (fCounter%4 == 0)
	tickMark = "[|]\r";
      else if (fCounter%4 == 1)
	tickMark = "[/]\r";
      else if (fCounter%4 == 2)
	tickMark = "[-]\r";
      else
	tickMark = "[\\]\r";
      fCounter++;
    }
    else
    {
      unsigned int counter = (unsigned int) std::fabs( progress/fGoal*100. );
      if (counter != fCounter)
      {
	fCounter = counter;
	std::stringstream s; s << "[";
	if (counter < 10)
	  s << "  ";
	else if (counter < 100)
	  s << " ";
	s << counter <<"%]  \r";
	tickMark = s.str();
      }
    }

    KEMField::cout << tickMark;
    KEMField::cout.flush();
  }

  void KTicker::EndTicker() const
  {
    KEMField::cout<<"[100%]"<<KEMField::endl;
  }
}
