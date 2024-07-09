
Formula Expressions
====================

The ability to calculate in-line formulas is another useful feature. The underlying implementation of the formula
processor relies on two external libraries. First, formulas are interpreted with the TinyExpr_ parser. This is a very
fast implementation that works for most simple expressions. If parsing fails, the formula is interpreted by the ROOT
TFormula_ class, which is slower but more versatile. To the user, the switching between both parsers is completely
transparent and no extra steps have to be taken.

In order to active the formula mode, the relevant expression must be enclosed in curly braces ``{...}``. Variables may
also be used within a formula, and all variable replacements will be done before the formula parsing (meaning that
the current value of the variable will be used in the formula.) An example of the formula syntax is given in the
following variable definition:

.. code-block:: xml

    <define name="my_variable" value="4.0"/>
    <define name="length" value="{2.3 + 2.0/sqrt([my_variable])}"/>
    <print name="length" value="[length]"/>

This example results in the variable ``length`` taking the value of 3.3.

Note that this example uses a standard function ``sqrt(x)`` that is supported by TinyExpr_. In general, any formulas
using advanced TMath_ functions or other complex syntax will use the TFormula_ parser. Simple TMath_ functions like
``TMath::Sqrt(x)`` or ``TMath::Sin(x)`` are mapped to their equivalent standard function (``sqrt(x)``, ``sin(x)``) that is
natively understood by TinyExpr_. The standard functions (and mathematical constants) are listed in the table below.

+---------------------------------------------------------------------------------------------------------+
| Standard functions and constants                                                                        |
+-------------+---------------+--------------------------+------------------------------------------------+
| XML syntax  | C++ function  | ROOT equivalent          | Description                                    |
+=============+===============+==========================+================================================+
| ``abs(x)``  | ``fabs(x)``   | ``TMath::Abs()``         | Compute absolute value.                        |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``acos(x)`` | ``acos(x)``   | ``TMath::ACos(x)``       | Compute arc cosine.                            |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``asin(x)`` | ``asin(x)``   | ``TMath::ASin(x)``       | Compute arc sine.                              |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``atan(x)`` | ``atan(x)``   | ``TMath::ATan(x)``       | Compute arc tangent.                           |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``atan2(x)``| ``atan2(x)``  | ``TMath::ATan2(x)``      | Compute arc tangent with two parameters.       |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``ceil(x)`` | ``ceil(x)``   | ``TMath::Ceil(x)``       | Round up value.                                |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``cos(x)``  | ``cos(x)``    | ``TMath::Cos(x)``        | Compute cosine.                                |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``cosh(x)`` | ``cosh(x)``   | ``TMath::CosH(x)``       | Compute hyperbolic cosine.                     |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``exp(x)``  | ``exp(x)``    | ``TMath::Exp(x)``        | Compute exponential function.                  |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``fac(x)``  |               | ``TMath::Factorial(x)``  | Compute factorial.                             |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``floor(x)``| ``floor(x)``  | ``TMath::Floor(x)``      | Round down value.                              |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``ln(x)``   | ``log(x)``    | ``TMath::Log(x)``        | Compute natural logarithm.                     |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``log(x)``  | ``log10(x)``  |                          | Compute common logarithm.                      |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``log10(x)``| ``log10(x)``  | ``TMath::Log10(x)``      | Compute common logarithm.                      |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``ncr(n,r)``|               | ``TMath::Binomial(n,r)`` | Compute combinations of `n` over `r`.          |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``npr(n,r)``|               |                          | Compute permuations of `n` over `r`.           |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``pow(x)``  | ``pow(x)``    | ``TMath::Pow(x)``        | Raise to power.                                |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``sin(x)``  | ``sin(x)``    | ``TMath::Sin(x)``        | Compute sine.                                  |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``sinh(x)`` | ``sinh(x)``   | ``TMath::SinH(x)``       | Compute hyperbolic sine.                       |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``sqrt(x)`` | ``sqrt(x)``   | ``TMath::Sqrt(x)``       | Compute square root.                           |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``tan(x)``  | ``tan(x)``    | ``TMath::Tan(x)``        | Compute tangent.                               |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``tanh(x)`` | ``tanh(x)``   | ``TMath::TanH(x)``       | Compute hyperbolic tangent.                    |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``e``       |               | ``TMath::Pi()``          | Fundamental constant.                          |
+-------------+---------------+--------------------------+------------------------------------------------+
| ``pi``      | ``M_PI``      | ``TMath::E()``           | Fundamental constant.                          |
+-------------+---------------+--------------------------+------------------------------------------------+
