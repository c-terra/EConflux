//2D mesh script for ResIPy (run the following in gmsh to generate a triangular mesh with topograpghy)
Mesh.Binary = 0;//specify we want ASCII format
cl=0.38;//define characteristic length
cl_factor= 2;//define characteristic length factor
//Define surface points
Point(1) = {-7.50,0.00,0.00,cl*cl_factor};//topography point
Point(2) = {0.00,0.00,0.00,cl};//electrode
Point(3) = {1.50,0.00,0.00,cl};//electrode
Point(4) = {3.00,0.00,0.00,cl};//electrode
Point(5) = {4.50,0.00,0.00,cl};//electrode
Point(6) = {6.00,0.00,0.00,cl};//electrode
Point(7) = {7.50,0.00,0.00,cl};//electrode
Point(8) = {9.00,0.00,0.00,cl};//electrode
Point(9) = {10.50,0.00,0.00,cl};//electrode
Point(10) = {12.00,0.00,0.00,cl};//electrode
Point(11) = {13.50,0.00,0.00,cl};//electrode
Point(12) = {15.00,0.00,0.00,cl};//electrode
Point(13) = {16.50,0.00,0.00,cl};//electrode
Point(14) = {18.00,0.00,0.00,cl};//electrode
Point(15) = {25.50,0.00,0.00,cl*cl_factor};//topography point
//construct lines between each surface point
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,9};
Line(9) = {9,10};
Line(10) = {10,11};
Line(11) = {11,12};
Line(12) = {12,13};
Line(13) = {13,14};
Line(14) = {14,15};
//add points below surface to make a fine mesh region
Point(16) = {-7.50,0.00,-5.50,cl*cl_factor};//base of smoothed mesh region
Point(17) = {0.75,0.00,-5.50,cl*cl_factor};//base of smoothed mesh region
Point(18) = {9.00,0.00,-5.50,cl*cl_factor};//base of smoothed mesh region
Point(19) = {17.25,0.00,-5.50,cl*cl_factor};//base of smoothed mesh region
Point(20) = {25.50,0.00,-5.50,cl*cl_factor};//base of smoothed mesh region
//make lines between base of fine mesh region points
Line(15) = {16,17};
Line(16) = {17,18};
Line(17) = {18,19};
Line(18) = {19,20};

//Adding boundaries
//end of boundaries.
//Add lines at leftmost side of fine mesh region.
Line(19) = {1,16};
//Add lines at rightmost side of fine mesh region.
Line(20) = {15,20};
//compile lines into a line loop for a mesh surface/region.
Line Loop(1) = {19, 15, 16, 17, 18, -20, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1};

//Background region (Neumann boundary) points
cln=18.75;//characteristic length for background region
Point(21) = {-97.50,0.00,0.00,cln};//far left upper point
Point(22) = {-97.50,0.00,-59.50,cln};//far left lower point
Point(23) = {115.50,0.00,0.00,cln};//far right upper point
Point(24) = {115.50,0.00,-59.50,cln};//far right lower point
//make lines encompassing all the background points - counter clock wise fashion
Line(21) = {1,21};
Line(22) = {21,22};
Line(23) = {22,24};
Line(24) = {24,23};
Line(25) = {23,15};
//Add line loops and plane surfaces for the Neumann region
Line Loop(2) = {21, 22, 23, 24, 25, 20, -18, -17, -16, -15, -19};
Plane Surface(1) = {1, 2};//Coarse mesh region surface

//Adding polygons
//end of polygons.
Plane Surface(2) = {1};//Fine mesh region surface

//Make a physical surface
Physical Surface(1) = {2, 1};

//End gmsh script
