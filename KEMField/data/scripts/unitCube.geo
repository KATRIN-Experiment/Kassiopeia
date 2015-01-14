scale = .05;

p1 = newp; Point(p1) = {0,0,0,scale};
p2 = newp; Point(p2) = {0,0,1,scale};
p3 = newp; Point(p3) = {0,1,0,scale};
p4 = newp; Point(p4) = {0,1,1,scale};
p5 = newp; Point(p5) = {1,0,0,scale};
p6 = newp; Point(p6) = {1,0,1,scale};
p7 = newp; Point(p7) = {1,1,0,scale};
p8 = newp; Point(p8) = {1,1,1,scale};

l1  = newreg; Line(l1) = {p1,p2};
l2  = newreg; Line(l2) = {p2,p4};
l3  = newreg; Line(l3) = {p4,p3};
l4  = newreg; Line(l4) = {p3,p1};
l5  = newreg; Line(l5) = {p5,p6};
l6  = newreg; Line(l6) = {p6,p8};
l7  = newreg; Line(l7) = {p8,p7};
l8  = newreg; Line(l8) = {p7,p5};
l9  = newreg; Line(l9) = {p1,p5};
l10 = newreg; Line(l10) = {p2,p6};
l11 = newreg; Line(l11) = {p4,p8};
l12 = newreg; Line(l12) = {p3,p7};

ll1 = newreg; Line Loop(ll1) = {l1,l2,l3,l4};
ll2 = newreg; Line Loop(ll2) = {l5,l6,l7,l8};
ll3 = newreg; Line Loop(ll3) = {l1,l10,-l5,-l9};
ll4 = newreg; Line Loop(ll4) = {l2,l11,-l6,-l10};
ll5 = newreg; Line Loop(ll5) = {l3,l12,-l7,-l11};
ll6 = newreg; Line Loop(ll6) = {l4,l9,-l8,-l12};

s1 = newreg; Ruled Surface(s1) = {ll1};
s2 = newreg; Ruled Surface(s2) = {ll2};
s3 = newreg; Ruled Surface(s3) = {ll3};
s4 = newreg; Ruled Surface(s4) = {ll4};
s5 = newreg; Ruled Surface(s5) = {ll5};
s6 = newreg; Ruled Surface(s6) = {ll6};
