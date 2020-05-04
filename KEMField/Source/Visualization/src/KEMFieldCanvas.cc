#include "KEMFieldCanvas.hh"

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

namespace KEMField
{
KEMFieldCanvas::KEMFieldCanvas(double x_1, double x_2, double y_1, double y_2, double zmir, bool isfull)
{
    full = isfull;
    x1 = x_1;
    x2 = x_2;
    y1 = y_1;
    y2 = y_2;

    zmirror = zmir;
}
}  // namespace KEMField
