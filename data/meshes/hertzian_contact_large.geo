SetFactory("OpenCASCADE");

h = 0.04;

// Substrate: wide/deep enough that the contact patch near x=0 is closer to a half-space.
Point(1) = {-4.0,  0.0, 0.0, h};
Point(2) = { 0.0,  0.0, 0.0, h};
Point(3) = { 4.0,  0.0, 0.0, h};
Point(4) = { 4.0, -3.0, 0.0, h};
Point(5) = {-4.0, -3.0, 0.0, h};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 1};

Curve Loop(1) = {1, 2, 3, 4, 5};
Plane Surface(1) = {1};

// Indenter: lower half disk, radius R = 1, tangent to the substrate at (0,0).
Point(10) = {-1.0, 1.0, 0.0, h};
Point(11) = { 0.0, 1.0, 0.0, h};
Point(12) = { 1.0, 1.0, 0.0, h};

Circle(10) = {10, 11, 2};
Circle(11) = {2, 11, 12};
Line(12) = {12, 10};

Curve Loop(2) = {10, 11, 12};
Plane Surface(2) = {2};

Physical Curve("indenter_top", 1) = {12};
Physical Curve("indenter_arc", 2) = {10, 11};
Physical Curve("block_top", 3) = {1, 2};
Physical Curve("block_bottom", 4) = {4};
Physical Curve("block_sides", 5) = {3, 5};
Physical Surface("block_body", 1) = {1};
Physical Surface("indenter_body", 2) = {2};

Mesh.Algorithm = 6;
Mesh.ElementOrder = 1;
